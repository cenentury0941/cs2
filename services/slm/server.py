"""FastAPI inference service: /health, /ready, /v1/decide."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schemas import DecideRequest, DecideResponse, Decision

logger = logging.getLogger(__name__)

_backend = os.environ.get("SLM_BACKEND", "mlx").strip().lower()
if _backend in ("cuda", "torch", "aws"):
    from inference_cuda import worker
else:
    from inference import worker


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting SLM worker load…")
    worker.load()
    yield


app = FastAPI(title="SimulCrowd SLM", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    if not worker.is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )
    return JSONResponse(content={"status": "ready"})


@app.post("/v1/decide", response_model=DecideResponse)
def decide(body: DecideRequest) -> DecideResponse:
    # Prefer one batched generate (CUDA). Mac mlx falls back to per-agent loop
    # via decide_many → decide_one.
    decide_many = getattr(worker, "decide_many", None)
    if callable(decide_many):
        decisions = decide_many(body.agents)
    else:
        decisions = [worker.decide_one(agent) for agent in body.agents]
    return DecideResponse(request_id=body.request_id, decisions=decisions)
