"""Background maintenance: drop sessions that have been idle for too long."""

from arclet.entari import plugin
from arclet.entari.event.lifespan import Startup
from arclet.entari.scheduler import every

from ..log import logger
from ..manager import LLMSessionManager
from ..service import LLMService


@plugin.listen(Startup)
async def cleanup_on_startup(_: LLMService):
    try:
        deleted = await LLMSessionManager.cleanup_expired()
        if deleted:
            logger.info(f"已清理 {deleted} 个过期会话")
    except Exception as e:
        logger.exception(f"清理过期会话失败: {e}")


@every(6, "hour", label="清理过期会话")
async def cleanup_periodically():
    try:
        deleted = await LLMSessionManager.cleanup_expired()
        if deleted:
            logger.info(f"已清理 {deleted} 个过期会话")
    except Exception as e:
        logger.exception(f"清理过期会话失败: {e}")
