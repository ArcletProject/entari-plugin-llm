from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agno.models.message import Message
from arclet.entari import ChannelType, Image, MessageChain
from arclet.letoderea import Contexts, waterfall
from entari_plugin_user import UserSession  # entari: plugin

from .capabilities import supports_vision
from .config import _conf, get_model_config
from .context import group_context
from .event import LLMCollectVariableEvent
from .locks import owner_lock_key, session_locks
from .service import llm
from .sessions import SessionInfo, resolve_session_owner


def _current_message_id(session: UserSession) -> str | None:
    message = getattr(session.internal.event, "message", None)
    return message.id if message is not None else None


class LLMSessionManager:
    @classmethod
    async def create_new_session(cls, user_id: int | str, model: str | None = None) -> SessionInfo:
        async with session_locks.acquire(owner_lock_key(user_id)):
            selected_model = model or get_model_config().name
            return await llm.session_store.create(user_id, selected_model)

    @classmethod
    async def switch(cls, user_id: int | str, session_id: str) -> bool:
        async with session_locks.acquire(owner_lock_key(user_id)):
            return await llm.session_store.activate(user_id, session_id)

    @classmethod
    async def delete(cls, user_id: int | str, session_id: str) -> bool:
        async with session_locks.acquire(owner_lock_key(user_id)):
            return await llm.session_store.delete(user_id, session_id)

    @classmethod
    async def get_current_session_info(cls, user_id: int | str) -> dict[str, Any] | None:
        async with session_locks.acquire(owner_lock_key(user_id)):
            session = await llm.session_store.current(user_id)
            return asdict(session) if session is not None else None

    @classmethod
    async def select_model(cls, user_id: int | str, model: str) -> bool:
        async with session_locks.acquire(owner_lock_key(user_id)):
            return await llm.session_store.set_model(user_id, model)

    @classmethod
    async def list_sessions(cls, user_id: int | str) -> list[SessionInfo]:
        async with session_locks.acquire(owner_lock_key(user_id)):
            return await llm.session_store.list(user_id)

    @classmethod
    async def chat(
        cls,
        user_prompt: MessageChain,
        *,
        session: UserSession,
        ctx: Contexts,
        model: str | None = None,
        new: bool = False,
    ) -> str:
        owner = resolve_session_owner(session)
        async with session_locks.acquire(owner_lock_key(owner)):
            return await cls._chat_locked(
                user_prompt,
                session=session,
                ctx=ctx,
                model=model,
                new=new,
                owner=owner,
            )

    @classmethod
    async def _chat_locked(
        cls,
        user_prompt: MessageChain,
        *,
        session: UserSession,
        ctx: Contexts,
        model: str | None,
        new: bool,
        owner: int | str,
    ) -> str:
        llm_session = None if new else await llm.session_store.current(owner)
        if llm_session is None:
            selected_model = model or get_model_config(None, session.channel.id).name
            llm_session = await llm.session_store.create(owner, selected_model)

        selected_model = llm_session.model or model or get_model_config(None, session.channel.id).name
        text = user_prompt.extract_plain_text()
        if session.channel_type == ChannelType.TEXT:
            context = group_context.consume(
                str(owner),
                current_id=_current_message_id(session),
                limit=_conf.group_context_limit,
            )
            if context:
                text = f"{context}\n\n[当前消息]\n{text}" if text else context

        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        if user_prompt.has(Image) and supports_vision(get_model_config(selected_model)):
            for image in list(user_prompt.select(Image)):
                content.append({"type": "image_url", "image_url": {"url": image.src}})
        user_message = Message(
            role="user",
            content=content,
            name=f"{session.user_name}({session.user_id})",
        )

        collect_event = LLMCollectVariableEvent(session.internal, llm_session, user_prompt)
        variables: dict[str, Any] = {}
        async for result in waterfall(collect_event, inherit_ctx=ctx):
            variables.update(result.value)

        response = await llm._generate_for_session(
            [user_message], variables, llm_session=llm_session, model=selected_model, session=session.internal, ctx=ctx
        )
        if response.stopped:
            final_answer = "[END_OF_RESPONSE]"
        else:
            final_answer = response.content or ""
        if not final_answer:
            return "对话失败，请稍后再试"
        return str(final_answer)
