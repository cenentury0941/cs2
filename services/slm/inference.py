"""Single mlx_lm worker: load once, generate constrained JSON / summaries."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Sequence

from config import MAX_TOKENS, MODEL_ID, SUMMARIZE_MAX_TOKENS, SUMMARIZE_TEMPERATURE, TEMPERATURE
from decision_parse import fallback, parse_output
from prompts import (
    SUMMARIZE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_summarize_user_prompt,
    build_user_prompt,
)
from schemas import AgentInput, Decision, SummarizeRequest, SummarizeResponse
from summarize_parse import fallback_summary, summarize_or_fallback

logger = logging.getLogger(__name__)


class MlxWorker:
    """One in-process mlx_lm model instance."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or os.environ.get("SLM_MODEL_ID", MODEL_ID)
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Download (if needed) and load the model into memory."""
        from mlx_lm import load

        self.model_id = os.environ.get("SLM_MODEL_ID", self.model_id)
        logger.info("Loading model %s …", self.model_id)
        self._model, self._tokenizer = load(self.model_id)
        self._ready = True
        logger.info("Model ready: %s", self.model_id)

    def decide_one(self, agent: AgentInput) -> Decision:
        """Run inference for one agent; never raise — always return a Decision."""
        try:
            raw = self._generate(
                SYSTEM_PROMPT,
                build_user_prompt(agent),
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            intent, destination = parse_output(raw, agent)
            return Decision(id=agent.id, intent=intent, destination=destination)
        except Exception:
            logger.exception("Inference failed for agent %s; using fallback", agent.id)
            return fallback(agent)

    def decide_many(self, agents: Sequence[AgentInput]) -> list[Decision]:
        """mlx path: still one generate per agent (no true batch API yet)."""
        return [self.decide_one(agent) for agent in agents]

    def summarize_one(self, req: SummarizeRequest) -> SummarizeResponse:
        """Narrative summary of Agent Info snapshot + events; never raise."""
        try:
            raw = self._generate(
                SUMMARIZE_SYSTEM_PROMPT,
                build_summarize_user_prompt(req),
                max_tokens=SUMMARIZE_MAX_TOKENS,
                temperature=SUMMARIZE_TEMPERATURE,
            )
            summary = summarize_or_fallback(raw, req)
        except Exception:
            logger.exception(
                "Summarize failed for agent %s; using fallback", req.agent_id
            )
            summary = fallback_summary(req.agent_id, req.snapshot, req.events)
        return SummarizeResponse(
            request_id=req.request_id,
            agent_id=req.agent_id,
            summary=summary,
        )

    def _generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if not self._ready or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not ready")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampler = make_sampler(temp=temperature)

        with self._lock:
            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                sampler=sampler,
            )
        return text if isinstance(text, str) else str(text)


worker = MlxWorker()
