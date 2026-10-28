from arclet.entari import declare_static, metadata
from arclet.entari.plugin import PluginRole

from .config import Config, _conf
from .event import LLMCollectVariableEvent as LLMCollectVariableEvent
from .event import LLMToolEvent as LLMToolEvent
from .tools import load_tools
from .tools import register as register_tool

metadata(
    name="LLM 工具箱",
    role=PluginRole.COMPLEX,
    author=[
        {"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"},
        {"name": "KomoriDev", "email": "mute231010@gmail.com"},
    ],
    version="0.5.0rc2",
    description="一个通用的 LLM 工具箱插件，提供了丰富的工具和模型配置选项，支持多种 LLM 模型。",
    urls={
        "homepage": "https://github.com/ArcletProject/entari-plugin-llm",
    },
    config=Config,
    readme="README.md",
)
declare_static()

from .capabilities import supports_vision as supports_vision
from .config import get_model_config as get_model_config
from .event import LLMResponseEvent as LLMResponseEvent
from .event import collect_vars as collect_vars
from .locks import SessionLocks as SessionLocks
from .locks import owner_lock_key as owner_lock_key
from .response import GenericResponse as GenericResponse
from .service import llm as llm
from .sessions import SessionInfo as SessionInfo
from .sessions import resolve_session_owner as resolve_session_owner

load_tools(_conf.tools)

from .handlers import chat as chat
from .handlers import check as check
from .handlers import command as command

__all__ = [
    "llm",
    "chat",
    "check",
    "command",
    "register_tool",
    "LLMCollectVariableEvent",
    "LLMToolEvent",
    "LLMResponseEvent",
    "collect_vars",
    "supports_vision",
    "get_model_config",
    "resolve_session_owner",
    "SessionInfo",
    "SessionLocks",
    "owner_lock_key",
    "GenericResponse",
]
