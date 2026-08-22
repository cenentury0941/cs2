"""FastAPI inference service: /health, /ready, /v1/decide, /v1/summarize."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schemas import DecideRequest, DecideResponse, SummarizeRequest, SummarizeResponse

logger = logging.getLogger(__name__)

_backend = os.environ.get("SLM_BACKEND", "mlx").strip().lower()
if _backend in ("cuda", "torch", "aws"):
    from inference_cuda import worker
else:
    from inference import worker

from openrouter import summarizer as openrouter_summarizer


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting SLM worker load…")
    worker.load()
    if openrouter_summarizer.is_configured:
        logger.info(
            "OpenRouter summarization ready (model=%s)",
            openrouter_summarizer.model,
        )
    else:
        logger.warning(
            "OpenRouter API key missing — /v1/summarize will use the local model"
        )
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


@app.post("/v1/summarize", response_model=SummarizeResponse)
def summarize(body: SummarizeRequest) -> SummarizeResponse:
    """Summarize Individual Agent Info via OpenRouter when configured."""
    if openrouter_summarizer.is_configured:
        return openrouter_summarizer.summarize(body)
    return worker.summarize_one(body)
