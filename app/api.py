import asyncio
import logging
import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.concurrency.limiter import (
    QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    limiter,
)
from app.dependencies import get_model_registry
from app.models.registry import ModelRegistry
from app.monitoring.metrics import observe_latency, record_request
from app.threadpool import get_executor

router = APIRouter()
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = Field(default="float", description="Only 'float' is supported")
    user: str | None = Field(default=None, description="OpenAI compatibility placeholder")


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = None


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"
    embedding_dimensions: int | None = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    models: list[str] | None = None


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    req: EmbeddingRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    request: Request,
) -> EmbeddingResponse:
    if req.encoding_format not in (None, "float"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'float' encoding_format is supported",
        )

    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    max_batch = int(os.getenv("MAX_BATCH_SIZE", "32"))
    if len(texts) > max_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch too large; max {max_batch} items",
        )

    max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "20000"))
    for idx, t in enumerate(texts):
        if len(t) > max_text_chars:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input at index {idx} exceeds max length {max_text_chars} chars",
            )

    start = time.perf_counter()
    try:
        async with limiter():
            try:
                model = registry.get(req.model)
            except KeyError as exc:
                record_request(req.model, "404")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {req.model} not found",
                ) from exc

            try:
                batcher = getattr(request.app.state, "batching_service", None)
                if batcher is not None and getattr(batcher, "enabled", False):
                    vectors = await batcher.enqueue(req.model, texts)
                else:
                    loop = asyncio.get_running_loop()
                    executor = get_executor()
                    vectors = await loop.run_in_executor(executor, model.embed, texts)
            except Exception as exc:  # pragma: no cover - unexpected runtime failure
                record_request(req.model, "500")
                logger.exception(
                    "embedding_failed",
                    extra={
                        "model": req.model,
                        "batch_size": len(texts),
                        "device": getattr(model, "device", None),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Embedding generation failed",
                ) from exc
    except QueueFullError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Request queue full",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    except ShuttingDownError as exc:
        record_request(req.model, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for worker",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc

    latency = time.perf_counter() - start
    observe_latency(req.model, latency)
    record_request(req.model, "200")
    logger.info(
        "embedding_request",
        extra={
            "model": req.model,
            "latency_ms": round(latency * 1000, 2),
            "batch_size": len(texts),
            "status": 200,
        },
    )

    data = [
        EmbeddingObject(index=i, embedding=vec.tolist()) for i, vec in enumerate(vectors)
    ]
    try:
        prompt_tokens = model.count_tokens(texts)
    except Exception:
        prompt_tokens = 0
    usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens, completion_tokens=None)
    return EmbeddingResponse(data=data, model=req.model, usage=usage)


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    registry: Annotated[ModelRegistry, Depends(get_model_registry)]
) -> ModelsResponse:
    models: list[ModelInfo] = []
    for name in registry.list_models():
        model = registry.get(name)
        dim = getattr(model, "dim", None)
        models.append(ModelInfo(id=name, embedding_dimensions=dim))
    return ModelsResponse(data=models)


@router.get("/health", response_model=HealthResponse)
async def health(
    registry: Annotated[ModelRegistry | None, Depends(get_model_registry, use_cache=False)] = None,
) -> HealthResponse:
    if registry is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model registry not initialized")
    try:
        models = registry.list_models()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Registry unavailable") from exc
    return HealthResponse(status="ok", models=models)
