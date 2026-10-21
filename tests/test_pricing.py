from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agno.metrics import RunMetrics

SRC = Path(__file__).resolve().parents[1] / "src"
if "entari_plugin_llm" not in sys.modules:
    package = types.ModuleType("entari_plugin_llm")
    package.__path__ = [str(SRC / "entari_plugin_llm")]
    sys.modules["entari_plugin_llm"] = package

from entari_plugin_llm._pricing import is_genai_prices_available, resolve_metrics_cost
from entari_plugin_llm.sessions import AgnoSessionStore

with patch("arclet.entari.add_service"):
    from entari_plugin_llm.service import LLMService


@dataclass
class FakeConfig:
    name: str
    provider: str | None = None
    api_key: str | None = "sk-test"
    base_url: str = "https://api.openai.com/v1"
    extra: dict[str, Any] = field(default_factory=dict)
    vision: bool | None = None


class MetricsCostTests(unittest.TestCase):
    def test_provider_reported_cost_wins(self):
        metrics = RunMetrics(input_tokens=1000, output_tokens=500, cost=0.123)
        self.assertEqual(resolve_metrics_cost(metrics, FakeConfig("gpt-4o", provider="openai")), 0.123)

    def test_zero_provider_reported_cost_is_kept(self):
        metrics = RunMetrics(input_tokens=1000, output_tokens=500, cost=0.0)
        self.assertEqual(resolve_metrics_cost(metrics, FakeConfig("gpt-4o", provider="openai")), 0.0)

    def test_cost_is_estimated_with_genai_prices_when_provider_omits_it(self):
        if not is_genai_prices_available():
            self.skipTest("genai-prices is not installed")
        from genai_prices import Usage, calc_price

        metrics = RunMetrics(input_tokens=1000, output_tokens=500)
        cost = resolve_metrics_cost(metrics, FakeConfig("gpt-4o", provider="openai"))
        expected = calc_price(Usage(input_tokens=1000, output_tokens=500), "gpt-4o").total_price
        self.assertIsInstance(cost, float)
        self.assertAlmostEqual(cost, float(expected), places=9)

    def test_unknown_model_returns_none(self):
        metrics = RunMetrics(input_tokens=10, output_tokens=5)
        self.assertIsNone(resolve_metrics_cost(metrics, FakeConfig("totally-unknown-model-xyz", provider="my-gateway")))

    def test_dict_metrics_are_supported(self):
        metrics = {"input_tokens": 1000, "output_tokens": 500, "cost": 0.01}
        self.assertEqual(resolve_metrics_cost(metrics, FakeConfig("gpt-4o", provider="openai")), 0.01)

    def test_missing_pricing_backend_returns_none(self):
        with patch("entari_plugin_llm._pricing._load_genai_prices", return_value=None):
            metrics = RunMetrics(input_tokens=1000, output_tokens=500)
            self.assertIsNone(resolve_metrics_cost(metrics, FakeConfig("gpt-4o", provider="openai")))

    def test_anthropic_provider_mapping(self):
        if not is_genai_prices_available():
            self.skipTest("genai-prices is not installed")
        metrics = RunMetrics(input_tokens=1000, output_tokens=500)
        cost = resolve_metrics_cost(metrics, FakeConfig("claude-sonnet-4-20250514", provider="anthropic"))
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0)


class SessionCostTests(unittest.TestCase):
    def test_session_cost_sums_reported_and_estimated_run_costs(self):
        if not is_genai_prices_available():
            self.skipTest("genai-prices is not installed")
        from genai_prices import Usage, calc_price

        runs = [
            SimpleNamespace(metrics=RunMetrics(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01)),
            SimpleNamespace(metrics=RunMetrics(input_tokens=1000, output_tokens=500, total_tokens=1500)),
        ]
        session = SimpleNamespace(
            summary=None,
            runs=runs,
            session_id="session-1",
            agent_data={},
            created_at=0,
            get_chat_history=lambda: [],
        )
        expected_estimate = float(calc_price(Usage(input_tokens=1000, output_tokens=500), "gpt-4o").total_price)
        store = AgnoSessionStore(None, "agent")  # type: ignore[arg-type]

        with (
            patch("entari_plugin_llm.sessions.get_session_model", return_value="gpt-4o"),
            patch("entari_plugin_llm.sessions.get_model_list", return_value={"gpt-4o"}),
            patch(
                "entari_plugin_llm.sessions.get_model_config",
                return_value=FakeConfig("gpt-4o", provider="openai"),
            ),
        ):
            info = store._to_info(session, 1, "session-1")  # type: ignore[arg-type]

        self.assertAlmostEqual(info.total_cost or 0.0, 0.01 + expected_estimate)


class UsageTrackingTests(unittest.TestCase):
    def setUp(self):
        self.service = LLMService()

    def _response(self, run_id: str, metrics: RunMetrics, paused: bool = False):
        return SimpleNamespace(run_id=run_id, metrics=metrics, is_paused=paused)

    def test_provider_cost_is_accumulated_across_continuation_segments(self):
        config = FakeConfig("gpt-4o", provider="openai")
        self.service._track_usage(
            self._response(
                "run-1", RunMetrics(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.10), paused=True
            ),
            config,
        )
        self.service._track_usage(
            self._response(
                "run-1", RunMetrics(input_tokens=200, output_tokens=50, total_tokens=250, cost=0.25), paused=True
            ),
            config,
        )
        self.service._track_usage(
            self._response("run-2", RunMetrics(input_tokens=50, output_tokens=50, total_tokens=100, cost=0.05)),
            config,
        )

        self.assertEqual(self.service.total_tokens, 350)
        self.assertAlmostEqual(self.service.total_cost, 0.30)
        self.assertEqual(self.service.total_calls, 3)

    def test_cost_is_estimated_when_provider_omits_it(self):
        if not is_genai_prices_available():
            self.skipTest("genai-prices is not installed")
        self.service._track_usage(
            self._response("run-1", RunMetrics(input_tokens=1000, output_tokens=500, total_tokens=1500)),
            FakeConfig("gpt-4o", provider="openai"),
        )
        self.assertGreater(self.service.total_cost, 0)

    def test_unknown_model_keeps_cost_at_zero(self):
        self.service._track_usage(
            self._response("run-1", RunMetrics(input_tokens=1000, output_tokens=500, total_tokens=1500)),
            FakeConfig("totally-unknown-model-xyz", provider="my-gateway"),
        )
        self.assertEqual(self.service.total_cost, 0.0)
