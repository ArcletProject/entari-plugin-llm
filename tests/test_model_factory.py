from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if "entari_plugin_llm" not in sys.modules:
    package = types.ModuleType("entari_plugin_llm")
    package.__path__ = [str(SRC / "entari_plugin_llm")]
    sys.modules["entari_plugin_llm"] = package

from entari_plugin_llm._capabilities import supports_vision
from entari_plugin_llm._litellm import is_litellm_available
from entari_plugin_llm.model_factory import (
    ModelBackendError,
    create_model,
    resolve_provider,
)


@dataclass
class FakeConfig:
    name: str
    provider: str | None = None
    api_key: str | None = "sk-test"
    base_url: str = "https://api.openai.com/v1"
    extra: dict[str, Any] = field(default_factory=dict)
    vision: bool | None = None


class ProviderResolutionTests(unittest.TestCase):
    def test_explicit_provider_wins(self):
        config = FakeConfig(
            "gpt-4o",
            provider="deepseek",
            base_url="https://api.siliconflow.cn/v1",
        )
        spec, model_id = resolve_provider(config)
        self.assertEqual(spec.key, "deepseek")
        self.assertEqual(model_id, "gpt-4o")

    def test_legacy_litellm_prefix(self):
        spec, model_id = resolve_provider(FakeConfig("deepseek/deepseek-chat"))
        self.assertEqual(spec.key, "deepseek")
        self.assertEqual(model_id, "deepseek-chat")

    def test_huggingface_style_model_id_is_not_mistaken_for_prefix(self):
        config = FakeConfig(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            base_url="https://api.siliconflow.cn/v1",
        )
        spec, model_id = resolve_provider(config)
        self.assertEqual(spec.key, "siliconflow")
        self.assertEqual(model_id, "Qwen/Qwen2.5-VL-72B-Instruct")

    def test_base_url_host_detection(self):
        config = FakeConfig("kimi-k2", base_url="https://api.moonshot.cn/v1")
        spec, model_id = resolve_provider(config)
        self.assertEqual(spec.key, "moonshot")
        self.assertEqual(model_id, "kimi-k2")

    def test_name_prefix_detection(self):
        spec, _ = resolve_provider(FakeConfig("claude-sonnet-4"))
        self.assertEqual(spec.key, "anthropic")

    def test_unknown_provider_is_litellm(self):
        spec, model_id = resolve_provider(FakeConfig("long-tail/model"))
        self.assertEqual(spec.key, "litellm")
        self.assertEqual(model_id, "long-tail/model")

    def test_explicit_unknown_provider_is_litellm(self):
        spec, model_id = resolve_provider(FakeConfig("foo", provider="my-gateway"))
        self.assertEqual(spec.key, "my-gateway")
        self.assertEqual(spec.kind, "litellm")
        self.assertEqual(model_id, "foo")


class ModelCreationTests(unittest.TestCase):
    def test_openai_extra_and_runtime_override(self):
        config = FakeConfig(
            "gpt-4o",
            extra={
                "temperature": 0.2,
                "top_p": 0.9,
                "custom_flag": 1,
                "request_params": {"seed": 42},
            },
        )
        model = create_model(config, request_params={"temperature": 0.3, "max_tokens": 100})

        self.assertEqual(type(model).__name__, "OpenAIChat")
        self.assertEqual(model.temperature, 0.3)
        self.assertEqual(model.max_tokens, 100)
        self.assertEqual(model.request_params, {"seed": 42, "custom_flag": 1})

    def test_openai_compatible_uses_openai_like(self):
        config = FakeConfig(
            "my-model",
            provider="openai-compatible",
            base_url="https://gateway.example.com/v1",
        )
        model = create_model(config)

        self.assertEqual(type(model).__name__, "OpenAILike")
        self.assertEqual(model.id, "my-model")
        self.assertEqual(model.base_url, "https://gateway.example.com/v1")
        self.assertEqual(model.api_key, "sk-test")

    def test_deepseek_native_adapter(self):
        model = create_model(FakeConfig("deepseek-chat"))
        self.assertEqual(type(model).__name__, "DeepSeek")
        self.assertEqual(model.id, "deepseek-chat")
        self.assertEqual(model.base_url, "https://api.deepseek.com")

    def test_siliconflow_preserves_model_id(self):
        config = FakeConfig(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            base_url="https://api.siliconflow.cn/v1",
        )
        model = create_model(config)
        self.assertEqual(type(model).__name__, "Siliconflow")
        self.assertEqual(model.id, "Qwen/Qwen2.5-VL-72B-Instruct")

    def test_zhipu_uses_compatible_endpoint(self):
        config = FakeConfig("glm-4.5", provider="zhipu")
        model = create_model(config)
        self.assertEqual(type(model).__name__, "OpenAILike")
        self.assertEqual(model.base_url, "https://open.bigmodel.cn/api/paas/v4")

    def test_gpt5_sampling_params_are_dropped(self):
        config = FakeConfig("gpt-5", extra={"temperature": 0.2})
        model = create_model(config, request_params={"top_p": 0.9})
        self.assertEqual(type(model).__name__, "OpenAIChat")
        self.assertIsNone(model.temperature)
        self.assertIsNone(model.top_p)

    def test_long_tail_uses_litellm(self):
        if not is_litellm_available():
            self.skipTest("LiteLLM is not installed")
        model = create_model(FakeConfig("some-long-tail/model"))
        self.assertEqual(type(model).__name__, "LiteLLM")
        self.assertEqual(model.id, "some-long-tail/model")

    def test_missing_native_sdk_falls_back_to_litellm(self):
        if not is_litellm_available():
            self.skipTest("LiteLLM is not installed")
        with patch("entari_plugin_llm.model_factory._model_class", side_effect=ImportError("missing sdk")):
            model = create_model(FakeConfig("claude-sonnet-4"))
        self.assertEqual(type(model).__name__, "LiteLLM")
        self.assertEqual(model.id, "anthropic/claude-sonnet-4")

    def test_missing_native_sdk_without_litellm_raises_actionable_error(self):
        with (
            patch("entari_plugin_llm.model_factory._model_class", side_effect=ImportError("missing sdk")),
            patch("entari_plugin_llm._litellm.is_litellm_available", return_value=False),
        ):
            with self.assertRaises(ModelBackendError) as caught:
                create_model(FakeConfig("claude-sonnet-4"))
        message = str(caught.exception)
        self.assertIn("Anthropic", message)
        self.assertIn("entari-plugin-llm[anthropic]", message)


class VisionTests(unittest.TestCase):
    def test_explicit_flag_wins(self):
        self.assertTrue(supports_vision(FakeConfig("deepseek-chat", vision=True)))
        self.assertFalse(supports_vision(FakeConfig("gpt-4o", vision=False)))

    def test_common_vision_models(self):
        for name in ("gpt-4o", "claude-sonnet-4", "gemini-2.5-pro", "Qwen/Qwen2.5-VL-72B-Instruct"):
            with self.subTest(name=name):
                config = FakeConfig(name, provider="openai" if name == "gpt-4o" else None)
                if name.startswith("Qwen/"):
                    config.base_url = "https://api.siliconflow.cn/v1"
                self.assertTrue(supports_vision(config))

    def test_text_only_models(self):
        for name, provider in (
            ("deepseek-chat", None),
            ("gpt-3.5-turbo", None),
            ("text-embedding-3-large", None),
            ("o1-mini", None),
            ("llama3.2", None),
        ):
            with self.subTest(name=name):
                self.assertFalse(supports_vision(FakeConfig(name, provider=provider)))


if __name__ == "__main__":
    unittest.main()
