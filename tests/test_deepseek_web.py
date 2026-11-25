from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

SRC = Path(__file__).resolve().parents[1] / "src"
if "entari_plugin_llm" not in sys.modules:
    package = types.ModuleType("entari_plugin_llm")
    package.__path__ = [str(SRC / "entari_plugin_llm")]
    package.register_tool = lambda func=None, **kwargs: func if func is not None else (lambda f: f)
    sys.modules["entari_plugin_llm"] = package

from entari_plugin_llm.builtins.tools import deepseek_web
from entari_plugin_llm.builtins.tools.deepseek_web import Config

SEARCH_RESPONSE = {
    "results": [
        {
            "rerank_score": 2.4,
            "title": "DeepSeek V4 发布",
            "url": "https://example.com/deepseek-v4",
            "snippet": "短摘要",
            "summary": "长摘要内容",
            "content": "网页正文内容",
            "site_name": "Example News",
            "site_icon": "https://example.com/favicon.ico",
            "date": "2026-06-01T00:00:00+00:00",
            "authority": 3,
        }
    ]
}


def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class DeepSeekSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_uses_defaults_and_formats_results(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=SEARCH_RESPONSE)

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret", count=5)),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("DeepSeek V4")

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "https://api.deepseek.com/v1/tools/search")
        self.assertEqual(request.headers["authorization"], "Bearer secret")
        self.assertEqual(
            json.loads(request.content),
            {"query": "DeepSeek V4", "count": 5, "need_content": False},
        )
        self.assertEqual(
            result,
            "1. DeepSeek V4 发布\n"
            "   URL: https://example.com/deepseek-v4\n"
            "   摘要: 长摘要内容\n"
            "   发布时间: 2026-06-01T00:00:00+00:00\n"
            "   权威度: 3",
        )

    async def test_search_arguments_override_config(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=SEARCH_RESPONSE)

        with (
            patch.object(
                deepseek_web,
                "_get_config",
                return_value=Config(api_key="secret", count=5, need_content=False),
            ),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("q", count=2, need_content=True)

        body = json.loads(requests[0].content)
        self.assertEqual(body["count"], 2)
        self.assertTrue(body["need_content"])
        self.assertIn("   正文:\n网页正文内容", result)

    async def test_search_reports_empty_results(self):
        def handler(request):
            return httpx.Response(200, json={"results": []})

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("q")

        self.assertEqual(result, "没有找到相关结果。")

    async def test_search_without_api_key_returns_hint(self):
        def handler(request):
            raise AssertionError("API must not be called without an api_key")

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config()),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("q")

        self.assertIn("api_key", result)

    async def test_api_error_is_returned_to_model(self):
        def handler(request):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "'count' must be between 1 and 50",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_request_error",
                    }
                },
            )

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("q", count=99)

        self.assertEqual(result, "DeepSeek API 返回错误（HTTP 400）：'count' must be between 1 and 50")

    async def test_invalid_json_response_is_returned_to_model(self):
        def handler(request):
            return httpx.Response(200, text="<html>not json</html>")

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("q")

        self.assertIn("无法解析", result)

    async def test_search_truncates_long_page_content(self):
        content = "正" * (deepseek_web.MAX_CONTENT_CHARS + 20)
        data = {"results": [{**SEARCH_RESPONSE["results"][0], "content": content}]}

        def handler(request):
            return httpx.Response(200, json=data)

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_search("q", need_content=True)

        self.assertIn("…（内容过长已截断）", result)
        self.assertNotIn(content, result)


class DeepSeekFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_returns_title_and_markdown(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"title": "示例文章", "content": "# 示例文章\n\n正文。"})

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_fetch("https://example.com/a")

        self.assertEqual(str(requests[0].url), "https://api.deepseek.com/v1/tools/fetch")
        self.assertEqual(json.loads(requests[0].content), {"url": "https://example.com/a"})
        self.assertEqual(result, "标题: 示例文章\n\n# 示例文章\n\n正文。")

    async def test_fetch_truncates_long_content(self):
        content = "正" * (deepseek_web.MAX_FETCH_CHARS + 20)

        def handler(request):
            return httpx.Response(200, json={"title": "长文", "content": content})

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_fetch("https://example.com/long")

        self.assertIn("…（内容过长已截断）", result)
        self.assertNotIn(content, result)

    async def test_fetch_reports_missing_content(self):
        def handler(request):
            return httpx.Response(200, json={"title": "空白页", "content": ""})

        with (
            patch.object(deepseek_web, "_get_config", return_value=Config(api_key="secret")),
            patch.object(deepseek_web, "_client", lambda: make_client(handler)),
        ):
            result = await deepseek_web.deepseek_fetch("https://example.com/empty")

        self.assertIn("未能提取网页正文", result)


if __name__ == "__main__":
    unittest.main()
