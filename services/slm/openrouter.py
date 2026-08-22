"""OpenRouter chat client used only for /v1/summarize."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import httpx

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_TIMEOUT_S,
    SUMMARIZE_MAX_TOKENS,
    SUMMARIZE_TEMPERATURE,
)
from prompts import SUMMARIZE_SYSTEM_PROMPT, build_summarize_user_prompt
from schemas import SummarizeRequest, SummarizeResponse
from summarize_parse import fallback_summary, summarize_or_fallback

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"(?:API\s*Key\s*:?\s*)?(sk-or-v1-[A-Za-z0-9]+)", re.I)


def normalize_openrouter_model(model: str) -> str:
    """Map ``openai.gpt-5.6-luna`` → ``openai/gpt-5.6-luna`` for the API."""
    raw = (model or "").strip()
    if not raw:
        return "openai/gpt-5.6-luna"
    if "/" in raw:
        return raw
    if "." in raw:
        provider, rest = raw.split(".", 1)
        return f"{provider}/{rest}"
    return raw


def resolve_api_key(explicit: Optional[str] = None) -> str:
    """Resolve key from arg, env, or repo ``openrouter_api_key.txt``."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = (OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if env:
        return env
    for path in _candidate_key_files():
        key = _read_key_file(path)
        if key:
            logger.info("Loaded OpenRouter API key from %s", path)
            return key
    return ""


def _candidate_key_files() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [
        here.parents[2],  # repo root (…/cs)
        here.parents[1],  # services/
        Path.cwd(),
    ]
    name = os.environ.get("OPENROUTER_API_KEY_FILE", "openrouter_api_key.txt")
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        path = (root / name).resolve() if not Path(name).is_absolute() else Path(name)
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _read_key_file(path: Path) -> str:
    try:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _KEY_RE.search(text)
    return match.group(1).strip() if match else text.strip()


class OpenRouterSummarizer:
    """Hosted chat completions for agent-info narratives only."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.model = normalize_openrouter_model(model or OPENROUTER_MODEL)
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.timeout_s = float(
            timeout_s if timeout_s is not None else OPENROUTER_TIMEOUT_S
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def summarize(self, req: SummarizeRequest) -> SummarizeResponse:
        if not self.is_configured:
            raise RuntimeError(
                "OpenRouter API key not configured "
                "(set OPENROUTER_API_KEY or openrouter_api_key.txt)"
            )
        try:
            raw = self._chat(
                system=SUMMARIZE_SYSTEM_PROMPT,
                user=build_summarize_user_prompt(req),
            )
            summary = summarize_or_fallback(raw, req)
        except Exception:
            logger.exception(
                "OpenRouter summarize failed for %s; using fallback", req.agent_id
            )
            summary = fallback_summary(req.agent_id, req.snapshot, req.events)
        return SummarizeResponse(
            request_id=req.request_id,
            agent_id=req.agent_id,
            summary=summary,
        )

    def _chat(self, *, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": SUMMARIZE_TEMPERATURE,
            "max_tokens": SUMMARIZE_MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/simulcrowd",
            "X-Title": "SimulCrowd Agent Summary",
        }
        with httpx.Client(timeout=self.timeout_s, trust_env=False) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some models return content parts
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter message content is not text")
        return content


summarizer = OpenRouterSummarizer()
