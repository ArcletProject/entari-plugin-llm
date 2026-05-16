from arclet.entari import declare_static, metadata, plugin

from .config import Config, _conf
from .log import _suppress_litellm_logging
from .tools import LLMToolEvent as LLMToolEvent

metadata(
    name="entari-plugin-llm",
    author=[
        {"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"},
        {"name": "KomoriDev", "email": "mute231010@gmail.com"},
    ],
    version="0.1.0",
    description="An Entari Plugin for LLM Chat with Function Call",
    config=Config,
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
