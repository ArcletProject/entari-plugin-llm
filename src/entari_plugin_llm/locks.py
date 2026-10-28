from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


def owner_lock_key(owner: int | str) -> str:
    """会话所有者键对应的锁键，避免用户 ID 与频道键意外撞车。"""
    return f"u:{owner}" if isinstance(owner, int) else f"c:{owner}"


class SessionLocks:
    """按会话所有者分片的异步锁，空闲时自动回收。"""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._counts: dict[str, int] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = asyncio.Lock()
            self._counts[key] = self._counts.get(key, 0) + 1

        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                count = self._counts.get(key, 1) - 1
                if count > 0:
                    self._counts[key] = count
                else:
                    self._counts.pop(key, None)
                    self._locks.pop(key, None)


session_locks = SessionLocks()
