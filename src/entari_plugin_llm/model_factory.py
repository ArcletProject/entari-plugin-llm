"""Build Agno model instances from the plugin's model configuration.

The factory prefers Agno's native provider adapters (OpenAI-compatible chat,
Anthropic, Gemini, DashScope, Bedrock, ...).  LiteLLM is only used as a lazy
fallback for providers that have no native adapter (or whose SDK is missing).
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from agno.models.base import Model
from agno.models.utils import MODEL_PROVIDER_CLASSES

logger = logging.getLogger("entari_plugin_llm.model_factory")

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

LITELLM_FALLBACK_HINT = (
    'Install the LiteLLM fallback with `pip install "entari-plugin-llm[litellm]"`, '
    "or configure a supported `provider`/`base_url`."
)


class ModelBackendError(RuntimeError):
    """Raised when the plugin cannot build a model backend from the configuration."""


class ModelConfigLike(Protocol):
    name: str
    provider: str | None
    api_key: str | None
    base_url: str
    extra: dict[str, Any]
    vision: bool | None


ProviderKind = Literal["openai", "compatible", "native", "litellm"]


@dataclass(frozen=True)
class ProviderSpec:
    """How a provider name maps to an Agno model class."""

    key: str
    kind: ProviderKind
    display: str
    aliases: tuple[str, ...] = ()
    agno_key: str | None = None
    default_base_url: str | None = None
    api_key_env: tuple[str, ...] = ()
    litellm_prefix: str | None = None
    extra_name: str | None = None
    requires_base_url: bool = False
    defaults: dict[str, Any] = field(default_factory=dict)


def _spec(key: str, kind: ProviderKind, display: str, **kwargs: Any) -> ProviderSpec:
    return ProviderSpec(key=key, kind=kind, display=display, **kwargs)


_PROVIDER_SPECS: dict[str, ProviderSpec] = {}
_PROVIDER_ALIASES: dict[str, str] = {}


def _register(spec: ProviderSpec) -> ProviderSpec:
    _PROVIDER_SPECS[spec.key] = spec
    _PROVIDER_ALIASES[spec.key] = spec.key
    for alias in spec.aliases:
        _PROVIDER_ALIASES[alias] = spec.key
    return spec


_register(
    _spec(
        "openai",
        "openai",
        "OpenAI",
        aliases=("openai-chat", "openai-chat-completions"),
        agno_key="openai-chat",
        default_base_url=OPENAI_DEFAULT_BASE_URL,
        api_key_env=("OPENAI_API_KEY",),
        litellm_prefix="openai",
    )
)
_register(
    _spec(
        "openai-compatible",
        "compatible",
        "OpenAI-compatible",
        aliases=(
            "openai-compat",
            "compatible",
            "gateway",
            "one-api",
            "new-api",
            "oneapi",
            "newapi",
            "localai",
            "xinference",
            "gpustack",
        ),
        litellm_prefix="openai",
        requires_base_url=True,
    )
)
_register(
    _spec(
        "anthropic",
        "native",
        "Anthropic",
        aliases=("claude",),
        agno_key="anthropic",
        api_key_env=("ANTHROPIC_API_KEY",),
        litellm_prefix="anthropic",
        extra_name="anthropic",
    )
)
_register(
    _spec(
        "google",
        "native",
        "Google Gemini",
        aliases=("gemini", "google-genai", "google-ai"),
        agno_key="google",
        api_key_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        litellm_prefix="gemini",
        extra_name="google",
    )
)
_register(
    _spec(
        "vertexai",
        "native",
        "Google Vertex AI",
        aliases=("vertex-ai", "vertex_ai", "vertex"),
        agno_key="google",
        api_key_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        litellm_prefix="vertex_ai",
        extra_name="google",
        defaults={"vertexai": True},
    )
)
_register(
    _spec(
        "dashscope",
        "native",
        "DashScope / Qwen",
        aliases=("qwen", "aliyun", "alibaba", "dashscope-intl"),
        agno_key="dashscope",
        api_key_env=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        litellm_prefix="dashscope",
    )
)
_register(
    _spec(
        "deepseek",
        "native",
        "DeepSeek",
        agno_key="deepseek",
        api_key_env=("DEEPSEEK_API_KEY",),
        litellm_prefix="deepseek",
    )
)
_register(
    _spec(
        "siliconflow",
        "native",
        "SiliconFlow",
        aliases=("silicon-flow", "siliconflow-cn"),
        agno_key="siliconflow",
        api_key_env=("SILICONFLOW_API_KEY",),
        litellm_prefix="siliconflow",
    )
)
_register(
    _spec(
        "moonshot",
        "native",
        "Moonshot / Kimi",
        aliases=("kimi",),
        agno_key="moonshot",
        api_key_env=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        litellm_prefix="moonshot",
    )
)
_register(
    _spec(
        "groq",
        "native",
        "Groq",
        agno_key="groq",
        api_key_env=("GROQ_API_KEY",),
        litellm_prefix="groq",
        extra_name="groq",
    )
)
_register(
    _spec(
        "openrouter",
        "native",
        "OpenRouter",
        agno_key="openrouter",
        api_key_env=("OPENROUTER_API_KEY",),
        litellm_prefix="openrouter",
    )
)
_register(
    _spec(
        "xai",
        "native",
        "xAI",
        aliases=("grok",),
        agno_key="xai",
        api_key_env=("XAI_API_KEY",),
        litellm_prefix="xai",
    )
)
_register(
    _spec(
        "together",
        "native",
        "Together AI",
        aliases=("together-ai",),
        agno_key="together",
        api_key_env=("TOGETHER_API_KEY",),
        litellm_prefix="together",
    )
)
_register(
    _spec(
        "fireworks",
        "native",
        "Fireworks AI",
        agno_key="fireworks",
        api_key_env=("FIREWORKS_API_KEY",),
        litellm_prefix="fireworks",
    )
)
_register(
    _spec(
        "perplexity",
        "native",
        "Perplexity",
        agno_key="perplexity",
        api_key_env=("PERPLEXITY_API_KEY",),
        litellm_prefix="perplexity",
    )
)
_register(
    _spec(
        "sambanova",
        "native",
        "SambaNova",
        agno_key="sambanova",
        api_key_env=("SAMBANOVA_API_KEY",),
        litellm_prefix="sambanova",
    )
)
_register(
    _spec(
        "cerebras",
        "native",
        "Cerebras",
        agno_key="cerebras",
        api_key_env=("CEREBRAS_API_KEY",),
        litellm_prefix="cerebras",
        extra_name="cerebras",
    )
)
_register(
    _spec(
        "nebius",
        "native",
        "Nebius",
        agno_key="nebius",
        api_key_env=("NEBIUS_API_KEY",),
        litellm_prefix="nebius",
    )
)
_register(
    _spec(
        "mistral",
        "native",
        "Mistral",
        agno_key="mistral",
        api_key_env=("MISTRAL_API_KEY",),
        litellm_prefix="mistral",
        extra_name="mistral",
    )
)
_register(
    _spec(
        "cohere",
        "native",
        "Cohere",
        agno_key="cohere",
        api_key_env=("COHERE_API_KEY",),
        litellm_prefix="cohere",
        extra_name="cohere",
    )
)
_register(
    _spec(
        "ollama",
        "native",
        "Ollama",
        agno_key="ollama",
        api_key_env=("OLLAMA_API_KEY",),
        litellm_prefix="ollama",
        extra_name="ollama",
    )
)
_register(
    _spec(
        "vllm",
        "native",
        "vLLM",
        agno_key="vllm",
        api_key_env=("VLLM_API_KEY",),
        litellm_prefix="vllm",
        requires_base_url=True,
    )
)
_register(
    _spec(
        "lmstudio",
        "native",
        "LM Studio",
        aliases=("lm-studio",),
        agno_key="lmstudio",
        litellm_prefix="lmstudio",
    )
)
_register(
    _spec(
        "zhipu",
        "compatible",
        "Zhipu / GLM",
        aliases=("glm", "bigmodel", "chatglm", "zhipuai"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env=("ZHIPU_API_KEY", "GLM_API_KEY"),
        litellm_prefix="zhipu",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "minimax",
        "compatible",
        "MiniMax",
        default_base_url="https://api.minimax.chat/v1",
        api_key_env=("MINIMAX_API_KEY",),
        litellm_prefix="minimax",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "volcengine",
        "compatible",
        "Volcengine / Doubao",
        aliases=("doubao", "ark"),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env=("ARK_API_KEY", "VOLCENGINE_API_KEY"),
        litellm_prefix="volcengine",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "hunyuan",
        "compatible",
        "Tencent Hunyuan",
        default_base_url="https://api.hunyuan.cloud.tencent.com/v1",
        api_key_env=("HUNYUAN_API_KEY",),
        litellm_prefix="hunyuan",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "stepfun",
        "compatible",
        "StepFun",
        aliases=("step",),
        default_base_url="https://api.stepfun.com/v1",
        api_key_env=("STEPFUN_API_KEY",),
        litellm_prefix="stepfun",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "yi",
        "compatible",
        "01.AI / Yi",
        aliases=("lingyiwanwu", "01ai"),
        default_base_url="https://api.lingyiwanwu.com/v1",
        api_key_env=("YI_API_KEY", "LINGYIWANWU_API_KEY"),
        litellm_prefix="yi",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "baichuan",
        "compatible",
        "Baichuan",
        default_base_url="https://api.baichuan-ai.com/v1",
        api_key_env=("BAICHUAN_API_KEY",),
        litellm_prefix="baichuan",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "sensenova",
        "compatible",
        "SenseNova",
        aliases=("sensechat",),
        default_base_url="https://api.sensenova.cn/compatible-mode/v1",
        api_key_env=("SENSENOVA_API_KEY",),
        litellm_prefix="sensenova",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "qianfan",
        "compatible",
        "Baidu Qianfan / ERNIE",
        aliases=("ernie", "baidu"),
        default_base_url="https://qianfan.baidubce.com/v2",
        api_key_env=("QIANFAN_API_KEY",),
        litellm_prefix="qianfan",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "spark",
        "compatible",
        "iFlytek Spark",
        aliases=("iflytek", "xfyun"),
        default_base_url="https://spark-api-open.xf-yun.com/v1",
        api_key_env=("SPARK_API_KEY", "IFLYTEK_API_KEY"),
        litellm_prefix="spark",
        requires_base_url=False,
    )
)
_register(
    _spec(
        "aws-bedrock",
        "native",
        "AWS Bedrock",
        aliases=("bedrock", "aws", "awsbedrock"),
        agno_key="aws-bedrock",
        litellm_prefix="bedrock",
        extra_name="aws-bedrock",
    )
)
_register(
    _spec(
        "litellm",
        "litellm",
        "LiteLLM",
        aliases=("litellm-openai",),
    )
)

_BASE_URL_PROVIDERS: dict[str, str] = {
    "api.openai.com": "openai",
    "api.deepseek.com": "deepseek",
    "api.siliconflow.cn": "siliconflow",
    "api.siliconflow.com": "siliconflow",
    "api.moonshot.ai": "moonshot",
    "api.moonshot.cn": "moonshot",
    "api.groq.com": "groq",
    "openrouter.ai": "openrouter",
    "api.x.ai": "xai",
    "api.together.xyz": "together",
    "api.fireworks.ai": "fireworks",
    "api.perplexity.ai": "perplexity",
    "api.sambanova.ai": "sambanova",
    "api.cerebras.ai": "cerebras",
    "api.mistral.ai": "mistral",
    "api.cohere.ai": "cohere",
    "api.cohere.com": "cohere",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "dashscope-intl.aliyuncs.com": "dashscope",
    "dashscope.aliyuncs.com": "dashscope",
    "open.bigmodel.cn": "zhipu",
    "api.minimax.chat": "minimax",
    "api.minimaxi.com": "minimax",
    "ark.cn-beijing.volces.com": "volcengine",
    "api.hunyuan.cloud.tencent.com": "hunyuan",
    "api.stepfun.com": "stepfun",
    "api.lingyiwanwu.com": "yi",
    "api.baichuan-ai.com": "baichuan",
    "api.sensenova.cn": "sensenova",
    "qianfan.baidubce.com": "qianfan",
    "spark-api-open.xf-yun.com": "spark",
}

_LOCAL_PORTS: dict[int, str] = {
    11434: "ollama",
    1234: "lmstudio",
}

_LEGACY_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "vertex_ai": "vertexai",
    "vertexai": "vertexai",
    "dashscope": "dashscope",
    "deepseek": "deepseek",
    "siliconflow": "siliconflow",
    "moonshot": "moonshot",
    "groq": "groq",
    "openrouter": "openrouter",
    "xai": "xai",
    "together": "together",
    "fireworks": "fireworks",
    "perplexity": "perplexity",
    "sambanova": "sambanova",
    "cerebras": "cerebras",
    "nebius": "nebius",
    "mistral": "mistral",
    "cohere": "cohere",
    "ollama": "ollama",
    "vllm": "vllm",
    "lmstudio": "lmstudio",
    "bedrock": "aws-bedrock",
    "zhipu": "zhipu",
    "minimax": "minimax",
    "volcengine": "volcengine",
    "hunyuan": "hunyuan",
    "stepfun": "stepfun",
    "yi": "yi",
    "baichuan": "baichuan",
    "sensenova": "sensenova",
    "qianfan": "qianfan",
    "spark": "spark",
    "litellm": "litellm",
}

_NAME_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("amazon.nova", "anthropic.", "meta.llama", "mistral.", "cohere.command"), "aws-bedrock"),
    (("gpt-", "chatgpt-", "o1", "o3", "o4"), "openai"),
    (("claude-",), "anthropic"),
    (("gemini-",), "google"),
    (("qwen", "qwq"), "dashscope"),
    (("deepseek",), "deepseek"),
    (("kimi", "moonshot"), "moonshot"),
    (("glm-", "glm4", "chatglm"), "zhipu"),
    (("grok-",), "xai"),
    (("mistral", "pixtral", "codestral", "magistral", "devstral", "ministral"), "mistral"),
    (("command-", "c4ai-"), "cohere"),
    (("minimax", "abab"), "minimax"),
    (("doubao-",), "volcengine"),
    (("hunyuan-",), "hunyuan"),
    (("step-",), "stepfun"),
    (("yi-",), "yi"),
    (("baichuan",), "baichuan"),
    (("sensenova", "sensechat"), "sensenova"),
    (("ernie-",), "qianfan"),
    (("spark-",), "spark"),
)

_RESERVED_EXTRA_KEYS = {
    "id",
    "name",
    "provider",
    "api_key",
    "base_url",
    "api_base",
    "request_params",
}

_UNSUPPORTED_REASONING_PARAMS = {"temperature", "top_p", "frequency_penalty", "presence_penalty"}


def _normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.strip().lower().replace("_", "-").replace(" ", "-")
    return normalized or None


def _lookup_provider(provider: str | None) -> ProviderSpec | None:
    key = _normalize_provider(provider)
    if not key:
        return None
    resolved = _PROVIDER_ALIASES.get(key, key)
    spec = _PROVIDER_SPECS.get(resolved)
    if spec is not None:
        return spec
    if resolved in MODEL_PROVIDER_CLASSES:
        return ProviderSpec(
            key=resolved,
            kind="native",
            display=resolved.replace("-", " ").title(),
            agno_key=resolved,
            litellm_prefix=resolved,
        )
    return None


def _is_default_base_url(base_url: str | None) -> bool:
    return not base_url or base_url.rstrip("/") == OPENAI_DEFAULT_BASE_URL


def _strip_provider_prefix(name: str, spec: ProviderSpec) -> str:
    for separator in ("/", ":"):
        if separator not in name:
            continue
        prefix, rest = name.split(separator, 1)
        rest = rest.strip()
        legacy_key = _LEGACY_PREFIXES.get(prefix.strip().lower())
        if rest and legacy_key == spec.key:
            return rest
    return name


def _spec_from_base_url(base_url: str | None) -> ProviderSpec | None:
    if _is_default_base_url(base_url):
        return None
    assert base_url is not None
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return _PROVIDER_SPECS["openai-compatible"]
    if hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        port = parsed.port
        if port in _LOCAL_PORTS:
            return _PROVIDER_SPECS[_LOCAL_PORTS[port]]
        return _PROVIDER_SPECS["openai-compatible"]
    for suffix, key in _BASE_URL_PROVIDERS.items():
        if hostname == suffix or hostname.endswith(f".{suffix}"):
            return _PROVIDER_SPECS[key]
    return _PROVIDER_SPECS["openai-compatible"]


def _spec_from_name(name: str) -> ProviderSpec | None:
    lowered = name.strip().lower()
    for prefixes, key in _NAME_PREFIXES:
        if lowered.startswith(prefixes):
            return _PROVIDER_SPECS[key]
    return None


def resolve_provider(model_config: ModelConfigLike) -> tuple[ProviderSpec, str]:
    """Resolve the provider spec and the provider-native model id."""

    raw_name = (model_config.name or "").strip()
    if not raw_name:
        raise ModelBackendError("Model name is empty; set `models[].name` in the plugin config.")

    explicit_provider = getattr(model_config, "provider", None)
    if explicit_provider:
        explicit = _lookup_provider(explicit_provider)
        if explicit is not None:
            return explicit, _strip_provider_prefix(raw_name, explicit)
        normalized = _normalize_provider(explicit_provider) or str(explicit_provider).strip().lower()
        return (
            ProviderSpec(
                key=normalized,
                kind="litellm",
                display=normalized.replace("-", " ").title(),
                litellm_prefix=normalized,
            ),
            raw_name,
        )

    for separator in ("/", ":"):
        if separator not in raw_name:
            continue
        prefix, rest = raw_name.split(separator, 1)
        legacy_key = _LEGACY_PREFIXES.get(prefix.strip().lower())
        if legacy_key is not None and rest.strip():
            return _PROVIDER_SPECS[legacy_key], rest.strip()

    spec = _spec_from_base_url(getattr(model_config, "base_url", None))
    if spec is not None:
        return spec, raw_name

    spec = _spec_from_name(raw_name)
    if spec is not None:
        return spec, raw_name

    return _PROVIDER_SPECS["litellm"], raw_name


def _effective_base_url(spec: ProviderSpec, model_config: ModelConfigLike) -> str | None:
    configured = (getattr(model_config, "base_url", None) or "").strip()
    if configured and not _is_default_base_url(configured):
        return configured
    if spec.default_base_url:
        return spec.default_base_url
    if spec.key == "openai":
        return configured or OPENAI_DEFAULT_BASE_URL
    if spec.requires_base_url:
        raise ModelBackendError(
            f"Provider '{spec.display}' requires an explicit `base_url` "
            "(for example `base_url: https://your-gateway.example.com/v1`)."
        )
    return None


def _resolve_api_key(spec: ProviderSpec, model_config: ModelConfigLike) -> str | None:
    configured = getattr(model_config, "api_key", None)
    if configured:
        return configured
    for env_name in spec.api_key_env:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def _model_class(spec: ProviderSpec) -> type[Model]:
    if spec.kind == "openai":
        module_path, class_name = "agno.models.openai", "OpenAIChat"
    elif spec.kind == "compatible":
        module_path, class_name = "agno.models.openai.like", "OpenAILike"
    elif spec.kind == "native":
        if not spec.agno_key or spec.agno_key not in MODEL_PROVIDER_CLASSES:
            raise ModelBackendError(f"Unsupported Agno provider '{spec.agno_key or spec.key}'.")
        module_path, class_name = MODEL_PROVIDER_CLASSES[spec.agno_key]
    else:
        raise ModelBackendError(f"Provider '{spec.display}' does not have a native model class.")

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _init_field_names(model_class: type[Model]) -> set[str]:
    try:
        return {item.name for item in fields(model_class) if item.init}
    except TypeError:
        names: set[str] = set()
        for base in reversed(model_class.__mro__):
            names.update(getattr(base, "__annotations__", {}))
        return names


def _split_extra(
    model_class: type[Model],
    extra: dict[str, Any],
    request_params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    constructor: dict[str, Any] = {}
    api_params: dict[str, Any] = {}
    nested_request_params = extra.get("request_params")
    merged_request_params: dict[str, Any] = (
        dict(nested_request_params) if isinstance(nested_request_params, dict) else {}
    )

    init_fields = _init_field_names(model_class)
    for key, value in extra.items():
        if key == "request_params" or key in _RESERVED_EXTRA_KEYS or value is None:
            continue
        if key in init_fields:
            constructor[key] = value
        else:
            api_params[key] = value

    for key, value in (request_params or {}).items():
        if key in _RESERVED_EXTRA_KEYS or value is None:
            continue
        # Per-call parameters win over ``extra``.  Prefer the adapter's
        # constructor field when one exists; otherwise forward as request params.
        if key in init_fields:
            constructor[key] = value
        else:
            api_params[key] = value
    merged_request_params.update(api_params)
    merged_request_params = {key: value for key, value in merged_request_params.items() if value is not None}
    return constructor, merged_request_params or None


def _drop_unsupported_params(
    spec: ProviderSpec,
    model_id: str,
    constructor: dict[str, Any],
    request_params: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if spec.key != "openai":
        return constructor, request_params

    lowered = model_id.lower()
    if not lowered.startswith(("gpt-5", "o1", "o3", "o4")):
        return constructor, request_params

    dropped = sorted(_UNSUPPORTED_REASONING_PARAMS.intersection(constructor))
    if request_params:
        dropped.extend(sorted(_UNSUPPORTED_REASONING_PARAMS.intersection(request_params)))
    if not dropped:
        return constructor, request_params

    logger.warning(
        "Model %s does not support %s; the parameter(s) will be ignored.",
        model_id,
        ", ".join(f"`{name}`" for name in dropped),
    )
    constructor = {key: value for key, value in constructor.items() if key not in _UNSUPPORTED_REASONING_PARAMS}
    if request_params:
        request_params = {
            key: value for key, value in request_params.items() if key not in _UNSUPPORTED_REASONING_PARAMS
        }
    return constructor, request_params


def _connection_kwargs(
    spec: ProviderSpec,
    model_id: str,
    model_class: type[Model],
    model_config: ModelConfigLike,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"id": model_id, **spec.defaults}
    init_fields = _init_field_names(model_class)

    api_key = _resolve_api_key(spec, model_config)
    if api_key is not None:
        kwargs["api_key"] = api_key
    elif spec.kind == "compatible":
        # OpenAILike defaults to the literal "not-provided", which prevents its
        # inherited OPENAI_API_KEY environment fallback.  Pass None explicitly.
        kwargs["api_key"] = None

    base_url = _effective_base_url(spec, model_config)
    if base_url:
        if "base_url" in init_fields:
            kwargs["base_url"] = base_url
        elif "api_base" in init_fields:
            kwargs["api_base"] = base_url
        elif "endpoint" in init_fields:
            kwargs["endpoint"] = base_url
        elif "host" in init_fields:
            kwargs["host"] = base_url.removesuffix("/v1").rstrip("/")
        elif "azure_endpoint" in init_fields:
            kwargs["azure_endpoint"] = base_url
        elif spec.key in {"anthropic", "cohere"}:
            client_params = dict(kwargs.get("client_params") or {})
            client_params.setdefault("base_url", base_url)
            kwargs["client_params"] = client_params
        else:
            logger.warning("Provider %s does not accept a custom base_url; ignoring %s.", spec.display, base_url)

    return kwargs


def _instantiate_native(
    spec: ProviderSpec,
    model_id: str,
    model_config: ModelConfigLike,
    request_params: dict[str, Any] | None,
) -> Model:
    model_class = _model_class(spec)
    kwargs = _connection_kwargs(spec, model_id, model_class, model_config)
    constructor_extra, api_params = _split_extra(model_class, model_config.extra, request_params)
    constructor_extra, api_params = _drop_unsupported_params(spec, model_id, constructor_extra, api_params)
    kwargs.update(constructor_extra)
    if api_params:
        kwargs["request_params"] = api_params
    return model_class(**kwargs)


def _litellm_model_id(spec: ProviderSpec, model_id: str) -> str:
    prefix = spec.litellm_prefix
    if not prefix or spec.kind == "litellm":
        return model_id
    if model_id.lower().startswith(f"{prefix.lower()}/"):
        return model_id
    return f"{prefix}/{model_id}"


def _instantiate_litellm(
    spec: ProviderSpec,
    model_id: str,
    model_config: ModelConfigLike,
    request_params: dict[str, Any] | None,
) -> Model:
    from ._litellm import require_litellm_model

    model_class = require_litellm_model()
    kwargs: dict[str, Any] = {"id": _litellm_model_id(spec, model_id)}

    base_url = _effective_base_url(spec, model_config)
    if base_url:
        kwargs["api_base"] = base_url

    api_key = _resolve_api_key(spec, model_config)
    if api_key is not None:
        kwargs["api_key"] = api_key

    constructor_extra, api_params = _split_extra(model_class, model_config.extra, request_params)
    if spec.key == "openai":
        constructor_extra, api_params = _drop_unsupported_params(spec, model_id, constructor_extra, api_params)
    kwargs.update(constructor_extra)
    if api_params:
        kwargs["request_params"] = api_params
    return model_class(**kwargs)


def _missing_dependency_message(spec: ProviderSpec, error: ImportError) -> str:
    missing = getattr(error, "name", None) or "the provider SDK"
    if spec.extra_name:
        install = f'pip install "entari-plugin-llm[{spec.extra_name}]"'
    elif spec.kind in {"openai", "compatible"}:
        install = "pip install openai"
    else:
        install = f"pip install {missing}"
    return (
        f"The {spec.display} backend requires `{missing}`. Install it with `{install}`, "
        f"or use the LiteLLM fallback: {LITELLM_FALLBACK_HINT}"
    )


def create_model(
    model_config: ModelConfigLike,
    request_params: dict[str, Any] | None = None,
) -> Model:
    """Create an Agno model for ``model_config``.

    Native adapters are preferred.  If the adapter's SDK is unavailable, the
    factory falls back to LiteLLM when it is installed; otherwise it raises an
    actionable :class:`ModelBackendError`.
    """

    spec, model_id = resolve_provider(model_config)

    if spec.kind == "litellm":
        return _instantiate_litellm(spec, model_id, model_config, request_params)

    try:
        return _instantiate_native(spec, model_id, model_config, request_params)
    except (ImportError, ModuleNotFoundError) as error:
        from ._litellm import is_litellm_available

        if is_litellm_available():
            logger.warning(
                "Native %s adapter is unavailable (%s); falling back to LiteLLM.",
                spec.display,
                error,
            )
            return _instantiate_litellm(spec, model_id, model_config, request_params)
        raise ModelBackendError(_missing_dependency_message(spec, error)) from error
