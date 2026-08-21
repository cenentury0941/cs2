"""SLM service configuration."""

from __future__ import annotations

import os

# mlx_lm default (Apple Silicon). Override with SLM_MODEL_ID.
MODEL_ID = os.environ.get(
    "SLM_MODEL_ID",
    "mlx-community/Phi-4-mini-instruct-4bit",
)
# transformers default (AWS Ubuntu / CUDA). Used when SLM_BACKEND=cuda
# and SLM_MODEL_ID is unset at process start before env is applied.
CUDA_MODEL_ID = os.environ.get(
    "SLM_CUDA_MODEL_ID",
    "microsoft/Phi-4-mini-instruct",
)
BACKEND = os.environ.get("SLM_BACKEND", "mlx").strip().lower()
HOST = os.environ.get("SLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SLM_PORT", "8080"))
TEMPERATURE = float(os.environ.get("SLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.environ.get("SLM_MAX_TOKENS", "100"))

ALLOWED_INTENTS = frozenset({"GO_TO_SEAT", "EXIT", "IDLE"})
