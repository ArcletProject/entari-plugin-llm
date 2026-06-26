from arclet.entari import declare_static, metadata, plugin
from arclet.entari.plugin import PluginRole

from .config import Config, _conf
from .log import _suppress_litellm_logging
from .tools import LLMToolEvent as LLMToolEvent

metadata(
    name="LLM 工具箱",
    role=PluginRole.COMPLEX,
    author=[
        {"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"},
        {"name": "KomoriDev", "email": "mute231010@gmail.com"},
    ],
    version="0.1.0",
    description="一个通用的 LLM 工具箱插件，提供了丰富的工具和模型配置选项，支持多种 LLM 模型，并且可以轻松集成到各种应用场景中。",
    urls={
        "homepage": "https://github.com/ArcletProject/entari-plugin-llm",
    },
    config=Config,
    readme="README.md",
)
declare_static()
_suppress_litellm_logging()

for tool in _conf.tools:
    plugin.load_plugin(tool)

from .handlers import chat as chat
from .handlers import check as check
from .handlers import command as command
from .service import llm as llm

__all__ = [
    "llm",
    "LLMToolEvent",
]
