"""Remove image parts from persisted history before a run.

Platform image URLs (notably QQ) expire. If such a URL is kept in the session
history, Agno re-downloads it on the next run and the provider rejects the whole
request. The current message is passed separately via ``run_input``, so only
history needs to be scrubbed.

When the model has already described an image (for example through the
``image_vision`` tool), the description is kept in ``image_memory`` and used
instead of a generic placeholder.
"""

from __future__ import annotations

from typing import Any

from .image_memory import image_description

MEDIA_PLACEHOLDER = "[历史图片已移除]"
_DESCRIPTION_LIMIT = 120
_MEDIA_TYPES = frozenset({"image_url", "input_image", "image"})


def strip_history_media(session: Any, run_response: Any = None, **_: Any) -> None:
    """Agno pre-hook: replace image parts in ``session.runs`` with text."""
    current_run_id = getattr(run_response, "run_id", None)
    for run in getattr(session, "runs", None) or []:
        if current_run_id is not None and getattr(run, "run_id", None) == current_run_id:
            continue
        for message in getattr(run, "messages", None) or []:
            _strip_message(message)


def _strip_message(message: Any) -> None:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        changed = False
        stripped: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in _MEDIA_TYPES:
                stripped.append({"type": "text", "text": _placeholder(part)})
                changed = True
            else:
                stripped.append(part)
        if changed:
            message.content = stripped or MEDIA_PLACEHOLDER

    images = getattr(message, "images", None)
    if isinstance(images, list) and images:
        message.images = []


def _placeholder(part: dict[str, Any]) -> str:
    description = image_description(_part_url(part))
    if not description:
        return MEDIA_PLACEHOLDER
    if len(description) > _DESCRIPTION_LIMIT:
        description = description[:_DESCRIPTION_LIMIT].rstrip() + "…"
    return f"[历史图片: {description}]"


def _part_url(part: dict[str, Any]) -> str | None:
    value = part.get("image_url")
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("url"), str):
        return value["url"]

    image = part.get("image")
    if isinstance(image, str):
        return image
    if isinstance(image, dict) and isinstance(image.get("url"), str):
        return image["url"]

    source = part.get("source")
    if isinstance(source, dict) and isinstance(source.get("url"), str):
        return source["url"]
    return None
