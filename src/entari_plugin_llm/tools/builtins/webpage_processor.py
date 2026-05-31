from arclet.entari import plugin
from launart import Launart
from entari_plugin_llm.log import logger
from entari_plugin_llm.tools import LLMToolEvent

try:
    from entari_plugin_browser import PlaywrightService
    browser_enabled = True
except ImportError:
    browser_enabled = False
    logger.error("未安装 entari_plugin_browser 插件，无法使用网页处理工具")


tools = plugin.dispatch(LLMToolEvent)


async def process_web_page(url: str) -> str | None:
    """处理网页内容

    Args:
        url: 网页 URL

    Returns:
        Optional[str]: 网页内容, 失败时返回 None
    """
    try:
        manager = Launart.current()
        pw_service = manager.get_component(PlaywrightService)
    except (LookupError, ValueError, RuntimeError):
        logger.error("PlaywrightService 未找到，无法处理网页内容")
        return None

    content_text = None
    async with pw_service.page() as page:
        try:
            await page.goto(url, timeout=60000)
        except Exception as e:
            logger.opt(exception=e).error(f"打开链接失败: {url}, 错误: {e}")
            return None

        if page_content := await page.query_selector("html"):
            content_text = await page_content.inner_text()

    return content_text

if browser_enabled:
    tools.register(process_web_page)
