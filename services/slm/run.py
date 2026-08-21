#!/usr/bin/env python3
"""Download/ensure the model and start the SLM HTTP service."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimulCrowd SLM service (mlx_lm)")
    parser.add_argument("--host", default=os.environ.get("SLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SLM_PORT", "8080")))
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SLM_MODEL_ID",
            "mlx-community/Phi-4-mini-instruct-4bit",
        ),
        help="HF / mlx model id",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="uvicorn auto-reload (dev only; reloads unload the model)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Must be set before uvicorn imports server → inference → config.
    os.environ["SLM_MODEL_ID"] = args.model
    os.environ["SLM_HOST"] = args.host
    os.environ["SLM_PORT"] = str(args.port)

    import uvicorn

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
