from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

_DEFAULT_KEEP = 200
_MAX_TEXT = 300


@dataclass(frozen=True, slots=True)
class BufferedMessage:
    message_id: str
    sender: str
    text: str
    created_at: datetime


class GroupContextBuffer:
    """按频道缓存尚未进入对话历史的群聊消息。

    只保留内存，进程重启后清空。消费时只取当前消息之前的消息，
    已经进入 Agno 历史的消息不会被重复注入。
    """

    def __init__(self, keep: int = _DEFAULT_KEEP) -> None:
        self._keep = keep
        self._items: dict[str, deque[BufferedMessage]] = {}

    def record(
        self,
        key: str,
        *,
        message_id: str,
        sender: str,
        text: str,
        created_at: datetime | None = None,
    ) -> None:
        if not message_id:
            return

        text = " ".join(text.split())
        if len(text) > _MAX_TEXT:
            text = f"{text[:_MAX_TEXT]}..."
        if not text:
            text = "[非文本消息]"

        items = self._items.get(key)
        if items is None:
            items = self._items[key] = deque(maxlen=self._keep)
        items.append(BufferedMessage(message_id, sender, text, created_at or datetime.now()))

    def consume(self, key: str, *, current_id: str | None, limit: int) -> str:
        """取出当前消息之前的最近 ``limit`` 条消息，并清理已消费的部分。"""
        if limit <= 0 or not current_id:
            return ""

        items = self._items.get(key)
        if not items:
            return ""

        messages = list(items)
        index = next((i for i, item in enumerate(messages) if item.message_id == current_id), None)
        if index is None:
            return ""

        selected = messages[:index][-limit:]
        remaining = messages[index + 1 :]
        if remaining:
            self._items[key] = deque(remaining, maxlen=self._keep)
        else:
            self._items.pop(key, None)

        if not selected:
            return ""

        lines = ["[群聊中的最近消息，仅供理解上下文]"]
        lines.extend(f"{item.created_at:%H:%M} sender {item.sender!r}: {item.text}" for item in selected)
        return "\n".join(lines)


group_context = GroupContextBuffer()
