#!/usr/bin/env python3
"""Start 1–N CUDA SLM workers plus a round-robin load balancer on AWS Ubuntu.

Client URL stays http://<host>:<port> (/health, /ready, /v1/decide).
Workers bind loopback on consecutive ports starting at --worker-base-port.
Default: 4 workers behind a round-robin LB (override with SLM_WORKERS).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"
_log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SimulCrowd SLM fleet on AWS Ubuntu (CUDA) with round-robin LB"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SLM_HOST", "0.0.0.0"),
        help="Public bind address for the load balancer (default 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SLM_PORT", "8080")),
        help="Public port clients use (default 8080)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLM_WORKERS", "4")),
        help="Number of identical CUDA worker processes (default 4)",
    )
    parser.add_argument(
        "--worker-base-port",
        type=int,
        default=int(os.environ.get("SLM_WORKER_BASE_PORT", "0")) or None,
        help="First worker loopback port (default: public port + 1)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SLM_MODEL_ID", _DEFAULT_MODEL),
        help="Hugging Face model id (CUDA)",
    )
    return parser.parse_args()


def _worker_ports(public_port: int, workers: int, base: int | None) -> List[int]:
    start = base if base is not None else public_port + 1
    ports = [start + i for i in range(workers)]
    if public_port in ports:
        raise SystemExit(
            f"Worker ports {ports} collide with public port {public_port}. "
            "Set --worker-base-port / SLM_WORKER_BASE_PORT higher."
        )
    return ports


def _spawn_worker(port: int, model: str, index: int, total: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["SLM_BACKEND"] = "cuda"
    env["SLM_MODEL_ID"] = model
    env["SLM_HOST"] = "127.0.0.1"
    env["SLM_PORT"] = str(port)
    # Avoid nested accidental LB / worker URL pollution in children.
    env.pop("SLM_WORKER_URLS", None)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    _log.info("Starting worker %d/%d on 127.0.0.1:%s", index + 1, total, port)
    return subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=env,
        start_new_session=True,
    )


def _terminate(procs: List[subprocess.Popen], sig: int = signal.SIGTERM) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                try:
                    p.send_signal(sig)
                except ProcessLookupError:
                    pass
    deadline = time.time() + 15.0
    for p in procs:
        remaining = max(0.1, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.workers < 1:
        raise SystemExit("--workers / SLM_WORKERS must be >= 1")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. On the Ubuntu instance run:\n"
            "  cd services/slm && python3 -m venv .venv-cuda && "
            "source .venv-cuda/bin/activate && "
            "pip install -r requirements-cuda.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Check nvidia-smi on the g6e.xlarge instance."
        )

    ports = _worker_ports(args.port, args.workers, args.worker_base_port)
    urls = [f"http://127.0.0.1:{p}" for p in ports]

    _log.info(
        "CUDA OK: %s (%.1f GiB). Fleet workers=%d public=http://%s:%s model=%s",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / (1024**3),
        args.workers,
        args.host,
        args.port,
        args.model,
    )
    if args.workers > 1:
        _log.warning(
            "Each worker loads a full model copy into GPU memory. "
            "Raise SLM_WORKERS only if VRAM allows."
        )

    procs: List[subprocess.Popen] = []

    def _shutdown(*_args: object) -> None:
        _log.info("Shutting down %d worker(s)…", len(procs))
        _terminate(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        for i, port in enumerate(ports):
            procs.append(_spawn_worker(port, args.model, i, args.workers))
            time.sleep(0.3)
            if procs[-1].poll() is not None:
                raise SystemExit(
                    f"Worker on port {port} exited early with code {procs[-1].returncode}"
                )

        os.environ["SLM_BACKEND"] = "cuda"
        os.environ["SLM_MODEL_ID"] = args.model
        os.environ["SLM_HOST"] = args.host
        os.environ["SLM_PORT"] = str(args.port)
        os.environ["SLM_WORKER_URLS"] = ",".join(urls)
        os.environ["SLM_WORKERS"] = str(args.workers)

        import uvicorn

        _log.info(
            "Load balancer on %s:%s → %s",
            args.host,
            args.port,
            os.environ["SLM_WORKER_URLS"],
        )
        uvicorn.run(
            "load_balancer:app",
            host=args.host,
            port=args.port,
            factory=False,
            log_level="info",
        )
    finally:
        _terminate(procs)


if __name__ == "__main__":
    main()
