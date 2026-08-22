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
# Narrative summaries need more headroom than tiny intent JSON.
SUMMARIZE_MAX_TOKENS = int(os.environ.get("SLM_SUMMARIZE_MAX_TOKENS", "256"))
SUMMARIZE_TEMPERATURE = float(os.environ.get("SLM_SUMMARIZE_TEMPERATURE", "0.4"))

# OpenRouter — used only for /v1/summarize (decide stays on local mlx/CUDA).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
# Accept either openai/gpt-5.6-luna or openai.gpt-5.6-luna
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai.gpt-5.6-luna").strip()
OPENROUTER_TIMEOUT_S = float(os.environ.get("OPENROUTER_TIMEOUT_S", "60"))

ALLOWED_INTENTS = frozenset({"GO_TO_SEAT", "EXIT", "IDLE"})
