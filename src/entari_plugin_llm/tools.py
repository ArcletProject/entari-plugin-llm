"""Bridge Entari LLMToolEvent subscribers to Agno-compatible callables.

Agno expects plain async functions. Entari tools are event subscribers that fire
through the `tools_pub` publisher with provider-injected dependencies. This module
wraps each registered tool so Agno can invoke it, while the Entari event machinery
handles DI and lifecycle.
"""

import asyncio
import importlib
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Any, overload

from agno.exceptions import AgentRunException
from agno.run.agent import RunOutput
from agno.run.requirement import RunRequirement
from agno.tools import Function, Toolkit
from agno.tools.function import _warn_hidden_media
from agno.utils.json_schema import get_json_schema
from agno.utils.schema import (
    AGNO_INJECTED_PARAMS,
    FRAMEWORK_INJECTED_PARAMS,
    is_bare_media_typed,
    is_schema_excluded,
)
from arclet.entari import Session, plugin
from arclet.entari.plugin.model import current_plugin
from arclet.letoderea import Propagator, Subscriber
from arclet.letoderea.context import Contexts, generate_contexts
from arclet.letoderea.exceptions import ExitState, _ExitException
from arclet.letoderea.scope import RegisterWrapper
from arclet.letoderea.subscriber import CompileParam
from docstring_parser import DocstringParam, parse
from tarina import Empty

from .config import _conf
from .event import JSON_TYPE, LLMToolEvent, tools_pub
from .log import logger
from .sessions import SessionInfo


@dataclass(frozen=True)
class RegisteredTool:
    subscriber: Subscriber[JSON_TYPE]
    function: Function
    plugin_id: str


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
            if plg := plugin.load_plugin(path):
                _configured_tool_plugins.add(plg.id)
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

        def dispose_tool(sub: Subscriber):
            current = available_functions.get(sub.__name__)
            if current is not None and current.subscriber is sub:
                available_functions.pop(sub.__name__)

        subscriber._attach_disposes(dispose_tool)
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
        available_functions[subscriber.__name__] = RegisteredTool(subscriber, function, plugin_id)  # type: ignore
        logger.debug(f"Registered tool: {subscriber.__name__}")
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
    if sub.__name__ not in available_functions:
        _ToolPropagator({"external_execution": True}).validate(sub)
    return True


def get_agno_tools() -> tuple[list[Toolkit | Function], dict[str, tuple[Callable, Function]]]:
    tools: list[Toolkit | Function] = []
    for path, toolkit_type in agno_tool_types.items():
        config = _conf.tools[path]
        options = {key: value for key, value in config.items() if not key.startswith("$")}

        try:
            toolkit = toolkit_type(**options)
        except ImportError as e:
            logger.warning(f"Agno tool {path} is unavailable: {e}")
            continue

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
    for name, registered in available_functions.items():
        # A removed tool configuration must not unload a plugin used elsewhere.
        if registered.plugin_id in _configured_tool_plugins and registered.plugin_id not in guards:
            continue
        fn = registered.function.model_copy()
        guard = guards.get(registered.plugin_id)

        async def execute(context, tool_args, sub=registered.subscriber, check=guard):
            if check is not None:
                check()
            logger.debug(f"Calling tool: {sub.__name__} with args: {tool_args}")
            return await sub.handle(context, inner=True)

        executors[name] = (execute, fn)
        tools.append(fn)
    return tools, executors


async def run_llm_tools(
    response: RunOutput,
    executors: dict[str, tuple[Callable, Function]],
    llm_session: SessionInfo | None = None,
    session: Session | None = None,
    ctx: Contexts | None = None,
):
    stops = []

    requirements = [requirement for requirement in response.active_requirements if requirement.needs_external_execution]

    async def execute_one(req: RunRequirement, context: Contexts):
        execution = req.tool_execution
        if not execution:
            req.set_external_execution_result("No tool execution found")
            return
        tool_name = execution.tool_name
        if not tool_name or tool_name not in executors:
            req.set_external_execution_result(f"Tool {tool_name} not found")
            return
        tool_args = execution.tool_args or {}
        context.update(tool_args)
        execute, fn = executors[tool_name]
        try:
            result = await execute(context, tool_args)
            if isinstance(result, ExitState):
                if result is ExitState.stop:
                    ans = json.dumps({"ok": True, "data": "已结束对话"}, ensure_ascii=False)
                else:
                    ans = json.dumps({"ok": True, "data": "Tool requested to stop the agent run"}, ensure_ascii=False)
            elif isinstance(result, _ExitException):
                if result.args[1]:
                    ans = json.dumps(
                        {"ok": False, "error": f"Tool requested to stop the agent run with message: {result.args[1]}"},
                        ensure_ascii=False,
                    )
                else:
                    ans = json.dumps({"ok": True, "data": result.args[0] if result.args else None}, ensure_ascii=False)
            elif isinstance(result, AgentRunException):
                ans = json.dumps({"ok": False, "error": f"AgentRunException: {str(result)}"}, ensure_ascii=False)
            elif isinstance(result, (str, int, float, bool, list, dict, type(None))):
                ans = json.dumps({"ok": True, "data": result}, ensure_ascii=False)
            else:
                ans = json.dumps({"ok": True, "data": str(result)}, ensure_ascii=False)
            req.set_external_execution_result(ans)
        except Exception as e:
            req.set_external_execution_result(json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False))
        if fn.stop_after_tool_call:
            stops.append(req)

    tool_ctx = await generate_contexts(LLMToolEvent(session=session, llm_session=llm_session), inherit_ctx=ctx)
    await asyncio.gather(*(execute_one(req, tool_ctx.copy()) for req in requirements))
    return not stops
