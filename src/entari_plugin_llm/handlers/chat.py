import random
from collections import deque

from arclet.entari import ChannelType, MessageChain, MessageCreatedEvent, Session, metadata
from arclet.entari.config import config_model_keys, config_model_validate
from arclet.entari.event.config import ConfigReload
from arclet.entari.event.send import SendResponse
from arclet.letoderea import BLOCK, on
from arclet.letoderea.context import Contexts
from entari_plugin_user import UserSession  # entari: plugin

from ..config import Config, _conf
from ..context import group_context
from ..exception import ModelNotFoundError
from ..manager import LLMSessionManager
from ..tools import load_tools
from .check import check_config

RECORD = deque(maxlen=16)

metadata(
    name="LLM 对话功能",
    author=[
        {"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"},
        {"name": "KomoriDev", "email": "mute231010@gmail.com"},
    ],
    version="0.1.0",
    description="LLM 工具箱插件的对话功能模块",
)


@on(SendResponse)
async def _record(event: SendResponse):
    if event.result and event.session:
        RECORD.append(event.session.event.sn)


@on(MessageCreatedEvent, priority=2, label="群聊上下文缓存")
async def buffer_group_message(session: Session, event: MessageCreatedEvent):
    """记录群聊消息，供后续对话作为上下文使用"""
    if _conf.group_context_limit <= 0 or session.channel.type != ChannelType.TEXT:
        return

    sender = (event.member.nick if event.member else None) or (event.user.nick or event.user.name or event.user.id)
    sender = sender if sender == event.user.id else f"{sender}({event.user.id})"
    group_context.record(
        f"{session.account.platform}:{session.channel.id}",
        message_id=event.message.id,
        sender=sender,
        text=session.content,
    )


@on(MessageCreatedEvent, priority=1000, label="AI-对话")
async def run_conversation(session: UserSession, ctx: Contexts, is_reply_me: bool = False, is_notice_me: bool = False):
    """利用 LLM 进行对话"""
    if session.internal.event.sn in RECORD:
        return BLOCK
    if is_reply_me or is_notice_me:
        pass
    elif random.random() < _conf.get_chance_of(session.session):
        pass
    else:
        return BLOCK
    try:
        answer = await LLMSessionManager.chat(
            session.elements,
            session=session,
            ctx=ctx,
        )
        if answer != "[END_OF_RESPONSE]":
            await session.send(answer)
    except ModelNotFoundError as e:
        await session.send(MessageChain(str(e)))
    except Exception as e:
        await session.send(MessageChain(str(e)))
    return BLOCK


@on(ConfigReload)
async def reload_config(event: ConfigReload):
    if event.scope != "plugin":
        return
    if event.key not in ("entari_plugin_llm", "llm"):
        return

    new_conf = config_model_validate(Config, event.value)
    load_tools(new_conf.tools)
    for key in config_model_keys(new_conf):
        setattr(_conf, key, getattr(new_conf, key))

    await check_config()
    return True
