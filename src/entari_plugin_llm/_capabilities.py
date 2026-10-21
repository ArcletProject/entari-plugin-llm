"""Best-effort vision-capability detection for native Agno adapters."""

from __future__ import annotations

from .model_factory import ModelBackendError, ModelConfigLike, resolve_provider

# Checked before the positive patterns: these model families are text-only
# (or produce non-chat output) even when their name contains a positive token.
_NON_VISION_PATTERNS: tuple[str, ...] = (
    "embedding",
    "embed-",
    "rerank",
    "whisper",
    "transcribe",
    "speech",
    "tts",
    "audio",
    "realtime",
    "moderation",
    "dall-e",
    "stable-diffusion",
    "image-generation",
    "flux",
    "sora",
    "o1-mini",
    "o1-preview",
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-coder",
    "gpt-3.5",
    "command-r",
    "command-light",
    "command-nightly",
)

_VISION_PATTERNS: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-4-turbo",
    "gpt-4-vision",
    "gpt-5",
    "chatgpt-4o",
    "o1",
    "o3",
    "o4",
    "claude-3",
    "claude-4",
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "gemini",
    "gemma-3",
    "grok-2-vision",
    "grok-3",
    "grok-4",
    "vision",
    "llava",
    "pixtral",
    "internvl",
    "minicpm-v",
    "omni",
    "vl-",
    "-vl",
    "_vl",
    "vl_",
    "vl2",
    "vl3",
    "mistral-small-3",
    "mistral-medium-3",
    "magistral",
    "glm-4v",
    "glm-4.1v",
    "glm-4.5v",
    "glm-4.6v",
    "kimi-latest",
    "kimi-k2.5",
    "nova-pro",
    "nova-lite",
    "nova-2",
    "llama-3.2-11b",
    "llama-3.2-90b",
    "llama-4",
    "llama4",
    "command-a-vision",
    "aya-vision",
    "deepseek-vl",
)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)


def supports_vision(model_config: ModelConfigLike) -> bool:
    """Return whether the configured model is likely to accept images.

    The explicit ``vision`` config flag always wins.  Otherwise this uses a
    conservative name-based heuristic because Agno has no unified capability
    registry.  Unknown models are assumed to be text-only; users can opt in
    with ``vision: true``.
    """

    explicit = getattr(model_config, "vision", None)
    if explicit is not None:
        return explicit

    try:
        _spec, model_id = resolve_provider(model_config)
    except ModelBackendError:
        model_id = getattr(model_config, "name", "") or ""

    name = model_id.strip().lower()
    if not name:
        return False
    if _matches_any(name, _NON_VISION_PATTERNS):
        return False
    return _matches_any(name, _VISION_PATTERNS)
