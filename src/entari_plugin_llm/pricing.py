from __future__ import annotations

import importlib
from typing import Any, Protocol

from .model_factory import resolve_provider

# genai-prices only knows a subset of provider ids. When a provider is missing
# here, leave the choice to genai-prices' model-name matching.
_GENAI_PRICES_PROVIDER_IDS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "vertexai": "google",
    "deepseek": "deepseek",
    "groq": "groq",
    "mistral": "mistral",
    "cohere": "cohere",
    "xai": "x-ai",
    "fireworks": "fireworks",
    "together": "together",
    "perplexity": "perplexity",
    "openrouter": "openrouter",
    "aws-bedrock": "aws",
}


class ModelConfigLike(Protocol):
    name: str
    provider: str | None
    base_url: str


def _load_genai_prices() -> Any | None:
    try:
        return importlib.import_module("genai_prices")
    except Exception:
        return None


def is_genai_prices_available() -> bool:
    return _load_genai_prices() is not None


def _metric_value(metrics: Any, name: str) -> Any:
    if isinstance(metrics, dict):
        return metrics.get(name)
    return getattr(metrics, name, None)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def resolve_metrics_cost(metrics: Any, model_config: ModelConfigLike | None) -> float | None:
    """Return the cost of a run, preferring the provider-reported value.

    Agno only fills ``RunOutput.metrics.cost`` when the provider returns a cost.
    For providers that do not, calculate an estimate with the optional
    ``genai-prices`` dataset. Unknown models and missing packages return
    ``None`` so cost tracking never breaks a chat run.
    """

    reported = _as_float(_metric_value(metrics, "cost"))
    if reported is not None:
        return reported

    if model_config is None:
        return None

    genai_prices = _load_genai_prices()
    if genai_prices is None:
        return None

    input_tokens = _as_int(_metric_value(metrics, "input_tokens"))
    output_tokens = _as_int(_metric_value(metrics, "output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None

    usage_kwargs: dict[str, int] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    cache_read_tokens = _as_int(_metric_value(metrics, "cache_read_tokens"))
    cache_write_tokens = _as_int(_metric_value(metrics, "cache_write_tokens"))
    if cache_read_tokens:
        usage_kwargs["cache_read_tokens"] = cache_read_tokens
    if cache_write_tokens:
        usage_kwargs["cache_write_tokens"] = cache_write_tokens

    try:
        spec, model_id = resolve_provider(model_config)
        calculation = genai_prices.calc_price(
            genai_prices.Usage(**usage_kwargs),
            model_id,
            provider_id=_GENAI_PRICES_PROVIDER_IDS.get(spec.key),
        )
    except Exception:
        return None
    return _as_float(calculation.total_price)
