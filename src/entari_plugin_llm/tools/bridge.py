"""Bridge Entari LLMToolEvent subscribers to Agno-compatible callables.

Agno expects plain async functions. Entari tools are event subscribers that fire
through the `tools_pub` publisher with provider-injected dependencies. This module
wraps each registered tool so Agno can invoke it, while the Entari event machinery
handles DI and lifecycle.
"""

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any
# from agno.tools import Function, tool
from agno.exceptions import AgentRunException, StopAgentRun
from arclet.entari import Session
from arclet.letoderea.context import Contexts, generate_contexts
from arclet.letoderea.exceptions import ExitState, _ExitException
from tarina import Empty

from ..log import logger
from .event import LLMToolEvent, available_functions


def _build_agno_tool(session: Session, ctx: Contexts, name: str) -> Callable[..., Awaitable[str]]:
    """Create an Agno-compatible async callable for a named Entari tool."""

    sub = available_functions[name]

    async def _wrapper(**kwargs: Any) -> str:
        tool_ctx = await generate_contexts(LLMToolEvent(session=session), inherit_ctx=ctx)
        tool_ctx.update(kwargs)
        logger.debug(f"Agno bridge calling tool: {name} with args: {kwargs}")

        try:
            resp = await sub.handle(tool_ctx, inner=True)
            if isinstance(resp, ExitState):
                if resp is ExitState.stop:
                    return json.dumps({"ok": True, "data": "已结束对话"}, ensure_ascii=False)
                raise StopAgentRun("Tool requested to stop the agent run")
            elif isinstance(resp, _ExitException):
                if resp.args[1]:
                    raise StopAgentRun(f"Tool requested to stop the agent run with message: {resp.args[1]}")
                result = {"ok": True, "data": resp.args[0] if resp.args else None}
                return json.dumps(result, ensure_ascii=False)
            elif isinstance(resp, AgentRunException):
                raise resp
            elif isinstance(resp, (str, int, float, bool, list, dict, type(None))):
                return json.dumps({"ok": True, "data": resp}, ensure_ascii=False)
            else:
                return json.dumps({"ok": True, "data": str(resp)}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"ok": False, "error": repr(e)}, ensure_ascii=False)

    # Copy metadata Agno uses for function schema
    _wrapper.__name__ = name
    _wrapper.__doc__ = sub.__doc__ or ""

    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {"return": str}
    for param in sub.params:
        if param.providers:
            continue
        annotations[param.name] = param.annotation
        parameters.append(
            inspect.Parameter(
                name=param.name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if param.default is Empty else param.default,
                annotation=param.annotation,
            )
        )
    _wrapper.__annotations__ = annotations
    setattr(_wrapper, "__signature__", inspect.Signature(parameters, return_annotation=str))

    return _wrapper


def get_agno_tools(session: Session, ctx: Contexts) -> list[Callable[..., Awaitable[str]]]:
    """Return all registered Entari tools as Agno-compatible callables."""
    return [_build_agno_tool(session, ctx, name) for name in available_functions]
