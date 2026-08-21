"""Shared parse / fallback for SLM workers (mlx or CUDA)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from config import ALLOWED_INTENTS
from schemas import AgentInput, Decision

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_output(raw: str, agent: AgentInput) -> Tuple[str, Optional[str]]:
    data = extract_json(raw)
    if data is None:
        logger.warning("No JSON in model output for %s: %r", agent.id, raw[:200])
        return fallback_fields(agent)

    intent = str(data.get("intent", "")).strip().upper()
    if intent not in ALLOWED_INTENTS:
        logger.warning("Invalid intent %r for %s", intent, agent.id)
        return fallback_fields(agent)

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


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
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


def fallback(agent: AgentInput) -> Decision:
    intent, destination = fallback_fields(agent)
    return Decision(id=agent.id, intent=intent, destination=destination)


def fallback_fields(agent: AgentInput) -> Tuple[str, Optional[str]]:
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
