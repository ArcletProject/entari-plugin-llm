"""Long-term memory helpers built on Agno's MemoryManager."""

from __future__ import annotations

from typing import Any

from agno.db.sqlite import AsyncSqliteDb
from agno.memory.manager import MemoryManager

from .config import MemoryConfig, get_model_config
from .model_factory import create_model


class LimitedMemoryManager(MemoryManager):
    """MemoryManager that only injects the most recently updated memories.

    Agno has no limit when adding memories to the system prompt. On busy group
    sessions that grows without bound, so trim it here while still keeping every
    memory in the database.
    """

    max_memories: int = 50

    def __init__(self, *args: Any, max_memories: int = 50, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_memories = max_memories

    def _trim(self, memories: list[Any] | None) -> list[Any]:
        memories = memories or []
        if self.max_memories <= 0 or len(memories) <= self.max_memories:
            return memories
        return memories[-self.max_memories :]

    def get_user_memories(self, user_id: str | None = None) -> list[Any]:
        return self._trim(super().get_user_memories(user_id=user_id))

    async def aget_user_memories(self, user_id: str | None = None) -> list[Any]:
        return self._trim(await super().aget_user_memories(user_id=user_id))


def build_memory_manager(db: AsyncSqliteDb | None, config: MemoryConfig) -> LimitedMemoryManager | None:
    if not config.enabled:
        return None

    model = create_model(get_model_config(config.model)) if config.model else None
    return LimitedMemoryManager(model=model, db=db, max_memories=config.max_memories)
