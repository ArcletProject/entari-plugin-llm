"""Execute Entari tools for paused Agno runs."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from agno.exceptions import AgentRunException
from agno.run.agent import RunOutput
from agno.run.requirement import RunRequirement
from agno.tools import Function
from arclet.entari import Session
from arclet.letoderea.context import Contexts, generate_contexts
from arclet.letoderea.exceptions import ExitState, _ExitException

from .event import LLMToolEvent
from .sessions import SessionInfo

ToolExecutors = dict[str, tuple[Callable, Function]]


async def run_llm_tools(
    response: RunOutput,
    executors: ToolExecutors,
    llm_session: SessionInfo | None = None,
    session: Session | None = None,
    ctx: Contexts | None = None,
) -> bool:
    """Execute paused Entari tools and return whether the Agent should continue."""
    stops: list[object] = []
    requirements: list[RunRequirement] = [
        requirement for requirement in response.active_requirements if requirement.needs_external_execution
    ]

    async def execute_one(requirement: RunRequirement, context: Contexts) -> None:
        execution = requirement.tool_execution
        if not execution:
            requirement.set_external_execution_result("No tool execution found")
            return

        tool_name = execution.tool_name
        if not tool_name or tool_name not in executors:
            requirement.set_external_execution_result(f"Tool {tool_name} not found")
            return

        tool_args = execution.tool_args or {}
        context.update(tool_args)
        execute, function = executors[tool_name]
        try:
            result = await execute(context, tool_args)
            requirement.set_external_execution_result(_serialize_result(result))
        except Exception as error:
            requirement.set_external_execution_result(
                json.dumps({"ok": False, "error": repr(error)}, ensure_ascii=False)
            )

        if function.stop_after_tool_call:
            stops.append(requirement)

    tool_context = await generate_contexts(
        LLMToolEvent(session=session, llm_session=llm_session),
        inherit_ctx=ctx,
    )
    await asyncio.gather(*(execute_one(requirement, tool_context.copy()) for requirement in requirements))
    return not stops


def _serialize_result(result: Any) -> str:
    if isinstance(result, ExitState):
        data = "已结束对话" if result is ExitState.stop else "Tool requested to stop the agent run"
        return json.dumps({"ok": True, "data": data}, ensure_ascii=False)
    if isinstance(result, _ExitException):
        if result.args[1]:
            return json.dumps(
                {"ok": False, "error": f"Tool requested to stop the agent run with message: {result.args[1]}"},
                ensure_ascii=False,
            )
        return json.dumps({"ok": True, "data": result.args[0] if result.args else None}, ensure_ascii=False)
    if isinstance(result, AgentRunException):
        return json.dumps({"ok": False, "error": f"AgentRunException: {result}"}, ensure_ascii=False)
    if isinstance(result, (str, int, float, bool, list, dict, type(None))):
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False)
    return json.dumps({"ok": True, "data": str(result)}, ensure_ascii=False)
