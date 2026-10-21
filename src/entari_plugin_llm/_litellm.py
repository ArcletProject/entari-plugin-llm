"""Lazy access to the optional LiteLLM fallback backend."""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agno.models.litellm import LiteLLM

INSTALL_HINT = 'LiteLLM backend is not installed. Install it with `pip install "entari-plugin-llm[litellm]"`.'


class LiteLLMUnavailableError(RuntimeError):
    """Raised when a feature needs LiteLLM but it is not installed."""


def is_litellm_available() -> bool:
    """Return whether the optional LiteLLM backend can be imported."""
    try:
        return find_spec("litellm") is not None
    except (ImportError, ValueError):
        return False


def require_litellm_model() -> type[LiteLLM]:
    """Return Agno's LiteLLM model adapter, with an actionable error."""
    # LiteLLM's logger is noisy on import; suppress it only when the fallback is used.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    try:
        from agno.models.litellm import LiteLLM
    except ImportError as exc:
        raise LiteLLMUnavailableError(INSTALL_HINT) from exc
    return LiteLLM
