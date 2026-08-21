#!/usr/bin/env python3
"""Start the CUDA SLM HTTP service on AWS Ubuntu (single process).

Client URL: http://<host>:<port> (/health, /ready, /v1/decide).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"
_log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SimulCrowd SLM service on AWS Ubuntu (CUDA)"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SLM_HOST", "0.0.0.0"),
        help="Bind address (default 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SLM_PORT", "8080")),
        help="Bind port (default 8080)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SLM_MODEL_ID", _DEFAULT_MODEL),
        help="Hugging Face model id (CUDA)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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

    # Must be set before uvicorn imports server → inference → config.
    os.environ["SLM_BACKEND"] = "cuda"
    os.environ["SLM_MODEL_ID"] = args.model
    os.environ["SLM_HOST"] = args.host
    os.environ["SLM_PORT"] = str(args.port)
    os.environ.pop("SLM_WORKER_URLS", None)

    _log.info(
        "CUDA OK: %s (%.1f GiB). Serving http://%s:%s model=%s",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / (1024**3),
        args.host,
        args.port,
        args.model,
    )

    import uvicorn

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        factory=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
