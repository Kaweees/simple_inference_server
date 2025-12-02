import asyncio
import importlib
from typing import Any

import pytest

from app.concurrency import limiter


async def _use_limiter(limiter_module: Any, delay: float = 0.05) -> None:
    async with limiter_module.limiter():
        await asyncio.sleep(delay)


def test_queue_full(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reduce limits to make the test quick and predictable
    monkeypatch.setenv("MAX_CONCURRENT", "1")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "2")
    monkeypatch.setenv("QUEUE_TIMEOUT_SEC", "0.2")

    importlib.reload(limiter)

    async def scenario() -> None:
        first = asyncio.create_task(_use_limiter(limiter, delay=0.1))
        await asyncio.sleep(0.01)  # ensure first holds the semaphore
        second = asyncio.create_task(_use_limiter(limiter, delay=0.01))
        await asyncio.sleep(0)  # let second enqueue

        with pytest.raises(limiter.QueueFullError):
            await _use_limiter(limiter, delay=0)

        await second
        await first

    asyncio.run(scenario())
