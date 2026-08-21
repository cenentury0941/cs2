"""Single mlx_lm worker: load once, generate constrained JSON."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional, Tuple

from config import ALLOWED_INTENTS, MAX_TOKENS, MODEL_ID, TEMPERATURE
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import AgentInput, Decision

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


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
            raw = self._generate(agent)
            intent, destination = self._parse_output(raw, agent)
            return Decision(id=agent.id, intent=intent, destination=destination)
        except Exception:
            logger.exception("Inference failed for agent %s; using fallback", agent.id)
            return self._fallback(agent)

    def _generate(self, agent: AgentInput) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if not self._ready or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not ready")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(agent)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampler = make_sampler(temp=TEMPERATURE)

        with self._lock:
            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                verbose=False,
                sampler=sampler,
            )
        return text if isinstance(text, str) else str(text)

    def _parse_output(
        self, raw: str, agent: AgentInput
    ) -> Tuple[str, Optional[str]]:
        data = self._extract_json(raw)
        if data is None:
            logger.warning("No JSON in model output for %s: %r", agent.id, raw[:200])
            return self._fallback_fields(agent)

        intent = str(data.get("intent", "")).strip().upper()
        if intent not in ALLOWED_INTENTS:
            logger.warning("Invalid intent %r for %s", intent, agent.id)
            return self._fallback_fields(agent)

        destination = data.get("destination")
        if destination is not None:
            destination = str(destination).strip() or None
            if destination and destination.lower() in ("null", "none"):
                destination = None

        if intent == "IDLE":
            destination = None
        elif intent == "EXIT" and not destination:
            destination = (agent.perception or {}).get("nearest_exit")
        elif intent == "GO_TO_SEAT" and not destination:
            destination = agent.assigned_seat or agent.goal

        return intent, destination

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _fallback(self, agent: AgentInput) -> Decision:
        intent, destination = self._fallback_fields(agent)
        return Decision(id=agent.id, intent=intent, destination=destination)

    @staticmethod
    def _fallback_fields(agent: AgentInput) -> Tuple[str, Optional[str]]:
        perception = agent.perception or {}
        if perception.get("fire_visible") or perception.get("smoke_visible"):
            return "EXIT", perception.get("nearest_exit")
        if perception.get("at_destination"):
            return "IDLE", None
        seat = agent.assigned_seat or agent.goal
        if seat and (
            perception.get("seat_known")
            or (isinstance(seat, str) and seat.startswith("Seat"))
        ):
            return "GO_TO_SEAT", seat
        if isinstance(agent.goal, str) and agent.goal.startswith("Exit"):
            return "EXIT", agent.goal
        return "IDLE", None


worker = MlxWorker()
