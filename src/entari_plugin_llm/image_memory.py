"""Remember what the model learned about images.

Platform image URLs (notably QQ) expire, so the URL itself is useless after a
while. If the model has already described an image, keep that description
around and reuse it when the image is stripped from the history.
"""

from __future__ import annotations

import json

from arclet.entari.localdata import local_data

_MAX_ENTRIES = 500
_MAX_URL_LENGTH = 2048
_cache: dict[str, str] | None = None


_store_path = local_data.get_data_file("entari_plugin_llm", "image_descriptions.json")


def _load() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache

    try:
        data = json.loads(_store_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _cache = {}
        return _cache

    images = data.get("images") if isinstance(data, dict) else None
    if isinstance(images, dict):
        _cache = {str(key): value for key, value in images.items() if isinstance(value, str)}
    else:
        _cache = {}
    return _cache


def _save() -> None:
    if _cache is None:
        return
    try:
        path = _store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"version": 1, "images": _cache}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        # The cache is best-effort; a read-only data dir must not break a run.
        pass


def reset_cache() -> None:
    """Drop the in-process cache (mainly useful for tests)."""
    global _cache
    _cache = None


def remember_image(url: str | None, description: str | None) -> None:
    """Store a model-generated description for an image URL."""
    if not url or not description or url.startswith("data:") or len(url) > _MAX_URL_LENGTH:
        return

    text = " ".join(description.split())
    if not text:
        return

    cache = _load()
    cache.pop(url, None)
    cache[url] = text
    while len(cache) > _MAX_ENTRIES:
        cache.pop(next(iter(cache)), None)
    _save()


def image_description(url: str | None) -> str | None:
    """Return the remembered description for an image URL, if any."""
    if not url:
        return None
    return _load().get(url)
