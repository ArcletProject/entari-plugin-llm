"""Bridge Entari LLMToolEvent subscribers to Agno-compatible callables.

Agno expects plain async functions. Entari tools are event subscribers that fire
through the `tools_pub` publisher with provider-injected dependencies. This module
wraps each registered tool so Agno can invoke it, while the Entari event machinery
handles DI and lifecycle.
"""

import importlib
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Any, overload

from agno.exceptions import AgentRunException
from agno.tools import Function, Toolkit
from agno.tools.function import _warn_hidden_media
from agno.utils.json_schema import get_json_schema
from agno.utils.schema import (
    AGNO_INJECTED_PARAMS,
    FRAMEWORK_INJECTED_PARAMS,
    is_bare_media_typed,
    is_schema_excluded,
)
from arclet.entari import plugin
from arclet.entari.plugin.model import Plugin, current_plugin
from arclet.letoderea import Propagator, Subscriber
from arclet.letoderea.scope import RegisterWrapper
from arclet.letoderea.subscriber import CompileParam
from docstring_parser import DocstringParam, parse
from tarina import Empty

from .config import _conf
from .event import JSON_TYPE, LLMToolEvent, tools_pub
from .log import logger
from .runtime import ToolExecutors


@dataclass(frozen=True)
class RegisteredTool:
    subscriber: Subscriber[JSON_TYPE]
    function: Function
    plugin_id: str
    owner: Plugin | None


available_functions: dict[str, RegisteredTool] = {}
agno_tool_types: dict[str, type[Toolkit]] = {}
_configured_tool_plugins: set[str] = set()


def make_call_guard(path: str, call_limit: int | None) -> Callable[[], None] | None:
    if call_limit is None:
        return None
    count = 0
    lock = Lock()

    def guard():
        nonlocal count
        with lock:
            if count >= call_limit:
                raise AgentRunException(f"Tool {path} reached its call limit ({call_limit}); use the existing results")
            count += 1

    return guard


def prepare_agno_functions(functions: Iterable[Function], guard: Callable[[], None] | None) -> None:
    def before_call(fc):
        if guard is not None:
            guard()
        logger.debug(f"Calling tool: {fc.function.name} with args: {fc.arguments or {}}")

    def wrap(previous):
        if previous is None:
            return before_call
        parameters = inspect.signature(previous).parameters

        def arguments(agent, team, run_context, fc):
            values = {"agent": agent, "team": team, "run_context": run_context, "fc": fc}
            return {key: value for key, value in values.items() if key in parameters}

        if inspect.iscoroutinefunction(previous):

            async def async_hook(agent=None, team=None, run_context=None, fc=None):
                before_call(fc)
                return await previous(**arguments(agent, team, run_context, fc))

            return async_hook

        def sync_hook(agent=None, team=None, run_context=None, fc=None):
            before_call(fc)
            return previous(**arguments(agent, team, run_context, fc))

        return sync_hook

    for function in functions:
        function.pre_hook = wrap(function.pre_hook)


def load_tools(configured_tools: dict[str, dict[str, Any]]) -> None:
    tool_types: dict[str, type[Toolkit]] = {}
    for path in configured_tools:
        if path.startswith("$"):
            continue
        if not path.startswith("agno."):
            if result := plugin.load_plugin(path, configured_tools[path]):
                _configured_tool_plugins.add(result.plugin.id)
            continue

        module_path, _, class_name = path.rpartition(".")
        try:
            tool_type = getattr(importlib.import_module(module_path), class_name, None)
        except ImportError as e:
            logger.warning(f"Agno tool {path} is unavailable: {e}")
            continue

        if not isinstance(tool_type, type) or not issubclass(tool_type, Toolkit):
            raise TypeError(f"{path} is not an Agno Toolkit")

        tool_types[path] = tool_type
        logger.debug(f"Registered tool: {path}")

    agno_tool_types.clear()
    agno_tool_types.update(tool_types)


def process_jsonschema(name: str, params: list[CompileParam], param_docs: list[DocstringParam]):
    # parameters = {"type": "object", "properties": {}, "required": []}
    type_hints = {param.name: param.annotation for param in params}
    if "agent" in type_hints:
        del type_hints["agent"]
    if "team" in type_hints:
        del type_hints["team"]
    if "run_context" in type_hints:
        del type_hints["run_context"]
    if "images" in type_hints:
        del type_hints["images"]
    if "videos" in type_hints:
        del type_hints["videos"]
    if "audios" in type_hints:
        del type_hints["audios"]
    if "files" in type_hints:
        del type_hints["files"]

    # Filter out return type and only process parameters
    excluded_params = ["return", "self", *FRAMEWORK_INJECTED_PARAMS]
    excluded_params.extend(param.name for param in params if param.name in AGNO_INJECTED_PARAMS)

    # Also exclude parameters whose types are framework-injected,
    # even if the parameter name differs (e.g. my_agent: Agent). See issue #6344.
    try:
        for param_name, hint in list(type_hints.items()):
            if is_schema_excluded(hint):
                del type_hints[param_name]
                excluded_params.append(param_name)
                if is_bare_media_typed(hint):
                    _warn_hidden_media(name, param_name)
    except Exception:
        pass

    param_type_hints = {param.name: type_hints.get(param.name) for param in params if param.name not in excluded_params}

    # Parse docstring for parameters
    param_descriptions = {}
    param_descriptions_clean = {}

    for param in param_docs:
        param_name = param.arg_name
        param_type = param.type_name

        if param_type:
            param_descriptions[param_name] = f"({param_type}) {param.description or ''}"
        else:
            param_descriptions[param_name] = param.description or ""
        param_descriptions_clean[param_name] = param.description or ""

    parameters = get_json_schema(type_hints=param_type_hints, param_descriptions=param_descriptions)
    parameters["required"] = [
        param.name for param in params if param.default is Empty and param.name not in excluded_params
    ]
    parameters["required"] = [name for name in parameters["required"] if name in parameters["properties"]]
    parameters["additionalProperties"] = False
    return parameters


class _ToolPropagator(Propagator):
    def __init__(self, tool_config: dict):
        self.tool_config = tool_config

    def validate(self, subscriber: Subscriber) -> bool:
        doc = inspect.cleandoc(subscriber.__doc__ or "")

        parsed = parse(doc)
        lines = []
        if parsed.short_description:
            lines.append(parsed.short_description)
        if parsed.long_description:
            lines.extend(parsed.long_description.split("\n"))

        entry_docs = "\n".join(lines)
        parameters = process_jsonschema(
            subscriber.__name__, [p for p in subscriber.params if not p.providers], parsed.params
        )

        tool_config = {
            "name": self.tool_config.get("name", subscriber.__name__),
            "description": self.tool_config.get("description", entry_docs),
            "instructions": self.tool_config.get("instructions"),
            "add_instructions": self.tool_config.get("add_instructions", True),
            "entrypoint": None,
            "cache_results": self.tool_config.get("cache_results", False),
            "cache_dir": self.tool_config.get("cache_dir"),
            "cache_ttl": self.tool_config.get("cache_ttl", 3600),
            "parameters": parameters,
            **{
                k: v
                for k, v in self.tool_config.items()
                if k
                not in [
                    "name",
                    "description",
                    "instructions",
                    "add_instructions",
                    "cache_results",
                    "cache_dir",
                    "cache_ttl",
                    "parameters",
                ]
                and v is not None
            },
        }

        # Automatically set show_result=True if stop_after_tool_call=True (unless explicitly set to False)
        if self.tool_config.get("stop_after_tool_call"):
            if "show_result" not in self.tool_config or self.tool_config.get("show_result") is None:
                tool_config["show_result"] = True

        function = Function(**tool_config)
        plg = current_plugin.get(None)
        plugin_id = plg.id if plg else subscriber.callable_target.__module__

        # Check for tool naming conflicts
        for registered in available_functions.values():
            if registered.function.name != function.name:
                continue
            if registered.owner is not None and not registered.owner.available:
                continue
            if (
                plg is not None
                and registered.owner is not None
                and registered.owner is not plg
                and registered.plugin_id == plugin_id
                and not plg.is_available
                and registered.owner.is_available
            ):
                continue
            raise ValueError(
                f"Tool name {function.name!r} is already registered by {registered.plugin_id}; "
                f"cannot register {plugin_id}"
            )

        def dispose_tool(sub: Subscriber):
            available_functions.pop(sub.id, None)

        subscriber._attach_disposes(dispose_tool)
        available_functions[subscriber.id] = RegisteredTool(subscriber, function, plugin_id, plg)  # type: ignore
        logger.debug(f"Registered tool: {function.name}")
        return False

    def compose(self):
        yield lambda: None


@overload
def register(func: Callable[..., Awaitable[Any]]) -> Subscriber[Awaitable[JSON_TYPE]]: ...


@overload
def register(
    *,
    name: str | None = None,
    description: str | None = None,
    instructions: str | None = None,
    add_instructions: bool = True,
    show_result: bool | None = None,
    stop_after_tool_call: bool | None = None,
    pre_hook: Callable | None = None,
    post_hook: Callable | None = None,
    tool_hooks: list[Callable] | None = None,
    cache_results: bool = False,
    cache_dir: str | None = None,
    cache_ttl: int = 3600,
) -> RegisterWrapper[JSON_TYPE, None]: ...


def register(*args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs["external_execution"] = True

    plg = plugin.get_plugin(1, optional=True)
    if plg:
        wpr = plg.dispatch(LLMToolEvent).handle()
    else:
        wpr = plugin.listen(LLMToolEvent)
    wpr.propagate(_ToolPropagator(kwargs))
    if len(args) == 1:
        return wpr(args[0])
    return wpr


@tools_pub.check
def _register_tool(_, sub: Subscriber):
    if sub.id not in available_functions:
        _ToolPropagator({"external_execution": True}).validate(sub)
    return True


def get_agno_tools() -> tuple[list[Toolkit | Function], ToolExecutors]:
    tools: list[Toolkit | Function] = []
    tool_owners: dict[str, str] = {}
    for path, toolkit_type in agno_tool_types.items():
        config = _conf.tools[path]
        options = {key: value for key, value in config.items() if not key.startswith("$")}

        try:
            toolkit = toolkit_type(**options)
        except ImportError as e:
            logger.warning(f"Agno tool {path} is unavailable: {e}")
            continue
        for function in toolkit.get_async_functions().values():
            if function.name in tool_owners:
                raise ValueError(
                    f"Tool name {function.name!r} conflicts between {tool_owners[function.name]} and {path}"
                )
            tool_owners[function.name] = path

        functions = {id(fn): fn for fn in (*toolkit.functions.values(), *toolkit.async_functions.values())}.values()
        guard = make_call_guard(path, config.get("$call_limit"))

        prepare_agno_functions(functions, guard)
        tools.append(toolkit)

    guards = {}
    for path, config in _conf.tools.items():
        if not path.startswith(("$", "agno.")):
            plg = plugin.find_plugin(path)
            guards[plg.id if plg else path] = make_call_guard(path, config.get("$call_limit"))

    executors: dict[str, tuple[Callable, Function]] = {}
    for registered in available_functions.values():
        if registered.owner is not None and not registered.owner.is_available:
            continue
        # A removed tool configuration must not unload a plugin used elsewhere.
        if registered.plugin_id in _configured_tool_plugins and registered.plugin_id not in guards:
            continue

        name = registered.function.name
        if name in tool_owners:
            raise ValueError(f"Tool name {name!r} conflicts between {tool_owners[name]} and {registered.plugin_id}")

        tool_owners[name] = registered.plugin_id
        fn = registered.function.model_copy()
        guard = guards.get(registered.plugin_id)

        async def execute(context, tool_args, sub=registered.subscriber, check=guard, tool_name=name):
            if check is not None:
                check()
            logger.debug(f"Calling tool: {tool_name} with args: {tool_args}")
            return await sub.handle(context, inner=True)

        executors[name] = (execute, fn)
        tools.append(fn)
    return tools, executors
