from __future__ import annotations

from typing import Any

import httpx
from arclet.entari import plugin
from arclet.entari.config import BasicConfModel

from entari_plugin_llm import register_tool
from entari_plugin_llm.log import logger

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
"""DeepSeek Search / Fetch API 的服务地址，固定为官方地址"""

SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"

MAX_CONTENT_CHARS = 4000
"""单条搜索结果的正文最多返回多少字符，避免超出模型上下文"""

MAX_FETCH_CHARS = 20000
"""单次抓取最多返回多少字符，避免超出模型上下文"""


class Config(BasicConfModel):
    api_key: str | None = None
    """DeepSeek API Key。需要单独配置，不会复用 models 中的 api_key"""
    count: int = 5
    """搜索默认返回的结果条数，范围 1～50"""
    need_content: bool = False
    """搜索时是否默认返回网页正文"""


class DeepSeekError(Exception):
    """DeepSeek API 调用失败"""


def _get_config() -> Config:
    try:
        return plugin.plugin_config(Config)
    except (LookupError, ValueError):
        return Config()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30)


def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text.strip() or "未知错误"
    return f"DeepSeek API 返回错误（HTTP {response.status_code}）：{str(message)[:500]}"


async def _post(path: str, payload: dict[str, Any], config: Config) -> dict[str, Any]:
    async with _client() as client:
        try:
            response = await client.post(
                f"{DEEPSEEK_BASE_URL}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        except httpx.HTTPError as e:
            raise DeepSeekError(f"请求 DeepSeek API 失败：{e}") from e

    if response.status_code != 200:
        raise DeepSeekError(_error_message(response))
    try:
        return response.json()
    except ValueError as e:
        raise DeepSeekError(f"无法解析 DeepSeek API 的响应：{e}") from e


def _require_api_key(config: Config) -> str | None:
    if config.api_key:
        return None
    return "未配置 DeepSeek API Key，请在 tools 的 ::deepseek_web 中设置 api_key"


def format_search_results(results: list[dict[str, Any]], need_content: bool) -> str:
    if not results:
        return "没有找到相关结果。"

    blocks: list[str] = []
    for index, item in enumerate(results, 1):
        lines = [
            f"{index}. {item.get('title') or '（无标题）'}",
            f"   URL: {item.get('url') or ''}",
        ]
        summary = item.get("summary") or item.get("snippet")
        if summary:
            lines.append(f"   摘要: {summary}")
        if date := item.get("date"):
            lines.append(f"   发布时间: {date}")
        authority = item.get("authority")
        if isinstance(authority, int) and authority >= 0:
            lines.append(f"   权威度: {authority}")
        if need_content and (content := (item.get("content") or "").strip()):
            if len(content) > MAX_CONTENT_CHARS:
                content = content[:MAX_CONTENT_CHARS] + "\n…（内容过长已截断）"
            lines.append("   正文:")
            lines.append(content)
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def format_fetch_result(data: dict[str, Any], url: str) -> str:
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()
    if not content:
        return f"未能提取网页正文：{title or url}"
    if len(content) > MAX_FETCH_CHARS:
        content = content[:MAX_FETCH_CHARS] + "\n…（内容过长已截断）"
    if title:
        return f"标题: {title}\n\n{content}"
    return content


@register_tool
async def deepseek_search(query: str, count: int | None = None, need_content: bool | None = None) -> str:
    """
    使用 DeepSeek 搜索 API 在互联网上搜索网页，返回标题、链接、摘要、发布时间与权威度。
    需要阅读网页正文时可把 need_content 设为 true，或随后用 deepseek_fetch 抓取指定链接。

    Args:
        query (str): 搜索词，支持 site:、-site:、引号、lang:、after:、before: 等搜索语法
        count (int | None): 返回条数，范围 1～50；未指定时使用配置中的默认值
        need_content (bool | None): 是否在结果中附带网页正文；未指定时使用配置中的默认值

    Returns:
        str: 整理后的搜索结果文本，失败时返回错误信息
    """
    config = _get_config()
    if error := _require_api_key(config):
        return error

    payload = {
        "query": query,
        "count": count if count is not None else config.count,
        "need_content": need_content if need_content is not None else config.need_content,
    }
    try:
        data = await _post(SEARCH_PATH, payload, config)
    except DeepSeekError as e:
        logger.warning(str(e))
        return str(e)
    return format_search_results(data.get("results") or [], bool(payload["need_content"]))


@register_tool
async def deepseek_fetch(url: str) -> str:
    """
    使用 DeepSeek Fetch API 抓取单个网页，返回网页标题与 Markdown 格式的正文。
    用于阅读搜索结果或用户给出的具体链接。

    Args:
        url (str): 要抓取的 HTTP/HTTPS 网页地址

    Returns:
        str: 网页标题与正文，失败时返回错误信息
    """
    config = _get_config()
    if error := _require_api_key(config):
        return error

    try:
        data = await _post(FETCH_PATH, {"url": url}, config)
    except DeepSeekError as e:
        logger.warning(str(e))
        return str(e)
    return format_fetch_result(data, url)
