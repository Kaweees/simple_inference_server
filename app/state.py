from __future__ import annotations

from app.batching import BatchingService
from app.models.registry import ModelRegistry

# Global holder for the loaded ModelRegistry instance.
model_registry: ModelRegistry | None = None
batching_service: BatchingService | None = None
