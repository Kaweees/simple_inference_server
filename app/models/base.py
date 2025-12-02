from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch


class EmbeddingModel(Protocol):
    name: str
    dim: int
    device: str | torch.device
    # Capabilities advertised by the handler, e.g., ["text-embedding"].
    capabilities: list[str]

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def count_tokens(self, texts: list[str]) -> int: ...
