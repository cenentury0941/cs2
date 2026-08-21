"""FastAPI inference service: /health, /ready, /v1/decide."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from inference import worker
from schemas import DecideRequest, DecideResponse, Decision

logger = logging.getLogger(__name__)


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
    decisions: list[Decision] = []
    for agent in body.agents:
        decisions.append(worker.decide_one(agent))
    return DecideResponse(request_id=body.request_id, decisions=decisions)
