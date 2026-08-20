from dataclasses import dataclass
from typing import Any, TypeAlias

from arclet.entari import MessageChain, MessageCreatedEvent, Session
from arclet.entari.const import ITEM_ACCOUNT, ITEM_SESSION
from arclet.letoderea import Contexts, Result, define, provide
from arclet.letoderea.provider import get_providers

from .sessions import SessionInfo


@dataclass
class LLMCollectVariableEvent:
    session: Session
    llm_session: SessionInfo
    user_message: MessageChain

    def check_result(self, value) -> Result[dict[str, Any]] | None:
        if isinstance(value, dict):
            return Result(value)
        return None


collect_vars = define(LLMCollectVariableEvent, name="llm/collect_vars")
collect_vars.providers.extend(
    [
        provide(SessionInfo, call="$llm_session"),
        provide(MessageChain, call="$user_message"),
    ]
)
collect_vars.providers.extend(get_providers(MessageCreatedEvent))


@collect_vars.gather
async def vars_gather(event: LLMCollectVariableEvent, context: Contexts):
    context[ITEM_ACCOUNT] = event.session.account
    context[ITEM_SESSION] = event.session
    context["$llm_session"] = event.llm_session
    context["$user_message"] = event.user_message


JSON_VALUE: TypeAlias = dict | list | str | int | float | bool | None
JSON_TYPE: TypeAlias = dict[str, "JSON_TYPE"] | list["JSON_TYPE"] | JSON_VALUE


@dataclass
class LLMToolEvent:
    session: Session | None
    llm_session: SessionInfo | None

    def check_result(self, value: Any) -> Result[JSON_TYPE] | None:
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            return Result(value)  # type: ignore


tools_pub = define(LLMToolEvent, name="tools_pub")
tools_pub.providers.append(provide(SessionInfo, call="$llm_session"))
tools_pub.providers.extend(get_providers(MessageCreatedEvent))


@tools_pub.gather
async def tools_gather(event: LLMToolEvent, context: Contexts):
    if event.session:
        context[ITEM_ACCOUNT] = event.session.account
        context[ITEM_SESSION] = event.session
    if event.llm_session:
        context["$llm_session"] = event.llm_session
