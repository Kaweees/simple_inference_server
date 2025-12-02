from fastapi import HTTPException, Request

from app.models.registry import ModelRegistry
from app.state import model_registry


def get_model_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "model_registry", None) or model_registry
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    return registry
