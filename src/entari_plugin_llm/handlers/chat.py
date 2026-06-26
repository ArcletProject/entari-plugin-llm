from collections import deque

from arclet.entari import MessageChain, MessageCreatedEvent, Session, filter_, metadata
from arclet.entari.config import config_model_validate
from arclet.entari.event.config import ConfigReload
from arclet.entari.event.send import SendResponse
from arclet.letoderea import BLOCK, on
from arclet.letoderea.context import Contexts

from ..config import Config, _conf
from ..exception import ModelNotFoundError
from ..manager import LLMSessionManager

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


@on(MessageCreatedEvent, priority=1000, label="AI 对话").if_(filter_.to_me)
async def run_conversation(session: Session, ctx: Contexts):
    """利用 LLM 进行对话"""
    if session.event.sn in RECORD:
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
    _conf.models = new_conf.models
    _conf.prompt = new_conf.prompt
    _conf.context_length = new_conf.context_length
    _conf.toolcall_max_steps = new_conf.toolcall_max_steps
