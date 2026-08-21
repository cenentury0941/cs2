"""Round-robin HTTP front for 1–N SLM worker processes.

Client-facing contract stays /health, /ready, /v1/decide.
Workers listen on loopback; this process is the public bind.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import threading
from typing import List, Optional
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = float(os.environ.get("SLM_LB_TIMEOUT", "120"))


def _parse_worker_urls() -> List[str]:
    raw = os.environ.get("SLM_WORKER_URLS", "").strip()
    if not raw:
        raise RuntimeError(
            "SLM_WORKER_URLS is empty. run_aws.py should set comma-separated "
            "http://127.0.0.1:<port> worker URLs before starting the load balancer."
        )
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        raise RuntimeError("SLM_WORKER_URLS parsed to zero URLs")
    return urls


class RoundRobin:
    def __init__(self, urls: List[str]) -> None:
        self._urls = list(urls)
        self._cycle = itertools.cycle(self._urls)
        self._lock = threading.Lock()

    @property
    def urls(self) -> List[str]:
        return list(self._urls)

    def next(self) -> str:
        with self._lock:
            return next(self._cycle)


workers = RoundRobin(_parse_worker_urls())
app = FastAPI(title="SimulCrowd SLM LB", version="0.1.0")


async def _probe(client: httpx.AsyncClient, base: str, path: str) -> Optional[httpx.Response]:
    try:
        return await client.get(urljoin(base + "/", path.lstrip("/")))
    except httpx.HTTPError as exc:
        logger.warning("Worker %s %s failed: %s", base, path, exc)
        return None


@app.get("/health")
async def health() -> dict:
    """LB process is up (does not require workers)."""
    return {"status": "ok", "workers": len(workers.urls)}


@app.get("/ready")
async def ready() -> JSONResponse:
    """200 once at least one worker has the model loaded."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        responses = await asyncio.gather(
            *[_probe(client, u, "/ready") for u in workers.urls]
        )
    ready_count = sum(1 for r in responses if r is not None and r.status_code == 200)
    payload = {
        "status": "ready" if ready_count else "not_ready",
        "workers_ready": ready_count,
        "workers_total": len(workers.urls),
    }
    if ready_count == 0:
        return JSONResponse(status_code=503, content=payload)
    return JSONResponse(content=payload)


async def _forward(method: str, path: str, request: Request) -> Response:
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    # Prefer a ready worker; fall through round-robin if probes fail.
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        tried: set[str] = set()
        last_exc: Optional[Exception] = None
        while len(tried) < len(workers.urls):
            base = workers.next()
            if base in tried:
                continue
            tried.add(base)
            probe = await _probe(client, base, "/ready")
            if probe is None or probe.status_code != 200:
                continue
            url = urljoin(base + "/", path.lstrip("/"))
            try:
                upstream = await client.request(
                    method, url, content=body, headers=headers
                )
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("Forward to %s failed: %s", url, exc)

        # No ready worker — try any worker once more (may still be loading).
        base = workers.next()
        url = urljoin(base + "/", path.lstrip("/"))
        try:
            upstream = await client.request(method, url, content=body, headers=headers)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )
        except httpx.HTTPError as exc:
            last_exc = exc

    logger.error("All workers failed for %s %s: %s", method, path, last_exc)
    return JSONResponse(
        status_code=503,
        content={"status": "unavailable", "detail": str(last_exc) if last_exc else "no workers"},
    )


@app.post("/v1/decide")
async def decide(request: Request) -> Response:
    return await _forward("POST", "/v1/decide", request)
