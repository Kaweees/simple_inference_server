from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.concurrency.limiter import MAX_CONCURRENT

_state: dict[str, ThreadPoolExecutor | None] = {"executor": None}


def get_executor() -> ThreadPoolExecutor:
    executor = _state["executor"]
    if executor is None:
        executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT,
            thread_name_prefix="embed-worker",
        )
        _state["executor"] = executor
    return executor


def shutdown_executor() -> None:
    executor = _state.get("executor")
    _state["executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)
