from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

import torch
from transformers import StoppingCriteria

from app.utils.device import resolve_device

logger = logging.getLogger(__name__)


def resolve_runtime_device(preference: str) -> str:
    """Resolve device preference to a concrete device string.

    Args:
        preference: Device preference string ("auto", "cpu", "cuda", "mps", etc.)

    Returns:
        Resolved device string

        Note:
        This function delegates to app.utils.device.resolve_device. Unlike the
        legacy behavior, non-auto values are validated so misconfigurations fail
        fast instead of silently falling back to CPU.
    """
    pref = preference or "auto"
    return resolve_device(pref, validate=pref != "auto")


def handle_oom(exc: BaseException, model_name: str, device: Any) -> None:
    """Handle CUDA OOM in a consistent, recoverable way.

    Logs the exception, clears CUDA cache if available, and re-raises.
    Chat batching and API layers will interpret the re-raised exception as a
    generic generation failure and surface a 500 error.

    Args:
        exc: The OOM exception
        model_name: Name of the model that triggered OOM
        device: Device the model is running on

    Raises:
        The original exception after cleanup
    """
    logger.exception(
        "chat_generate_oom",
        extra={
            "model": model_name,
            "device": str(device) if device else "unknown",
        },
    )
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    raise exc


class StopOnTokens(StoppingCriteria):
    """Stop generation when any of the provided token sequences is produced."""

    def __init__(self, stop_token_ids: list[list[int]]) -> None:
        super().__init__()
        # Keep only non-empty stop sequences
        self.stop_token_ids = [ids for ids in stop_token_ids if ids]
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: D401
        if not self.stop_token_ids:
            return False
        generated = input_ids[0].tolist()
        for ids in self.stop_token_ids:
            if len(ids) <= len(generated) and generated[-len(ids) :] == ids:
                self.triggered = True
                return True
        return False


class StopOnCancel(StoppingCriteria):
    """Stop generation when a cancellation event is set."""

    def __init__(self, event: threading.Event) -> None:
        super().__init__()
        self.event = event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: D401
        return self.event.is_set()


class StopOnCancelAny(StoppingCriteria):
    """Stop batched generation when any cancellation event is set."""

    def __init__(self, events: list[threading.Event]) -> None:
        super().__init__()
        self.events = events

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: D401
        return any(ev.is_set() for ev in self.events)


def trim_with_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
    """Trim the generated text at the earliest occurrence of any stop string."""

    if not stop:
        return text, False

    earliest_idx: int | None = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx

    if earliest_idx is None:
        return text, False

    return text[:earliest_idx].rstrip(), True


