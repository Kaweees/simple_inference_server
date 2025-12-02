import os
from contextlib import suppress

from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.applications import Starlette

REQUEST_COUNT = Counter(
    "embedding_requests_total",
    "Total number of embedding requests",
    labelnames=("model", "status"),
)

REQUEST_LATENCY = Histogram(
    "embedding_request_latency_seconds",
    "Embedding request latency in seconds",
    labelnames=("model",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

QUEUE_REJECTIONS = Counter(
    "embedding_queue_rejections_total",
    "Requests rejected due to queue limits",
)


def setup_metrics(app: Starlette) -> None:
    if os.getenv("ENABLE_METRICS", "1") == "0":
        return
    # Mount Prometheus ASGI app at /metrics
    app.mount("/metrics", make_asgi_app())


def record_request(model: str, status: str) -> None:
    with suppress(Exception):
        REQUEST_COUNT.labels(model=model, status=status).inc()


def observe_latency(model: str, seconds: float) -> None:
    with suppress(Exception):
        REQUEST_LATENCY.labels(model=model).observe(seconds)


def record_queue_rejection() -> None:
    with suppress(Exception):
        QUEUE_REJECTIONS.inc()
