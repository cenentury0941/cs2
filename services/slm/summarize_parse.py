"""Clean model output and rule fallback for /v1/summarize."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from schemas import SummarizeRequest


def clean_summary(raw: str) -> str:
    """Strip fences / quotes; return empty string if nothing usable."""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening fence and optional closing fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def fallback_summary(
    agent_id: str,
    snapshot: Optional[Dict[str, str]] = None,
    events: Optional[List[str]] = None,
) -> str:
    """Deterministic summary when the model fails or returns empty text."""
    snap = snapshot or {}
    intent = snap.get("intent") or "unknown"
    goal = snap.get("goal") or "—"
    fire = snap.get("fire_perceived")
    smoke = snap.get("smoke_perceived")
    parts = [
        f"{agent_id} currently has intent {intent} with goal {goal}."
    ]
    hazards = []
    if fire == "yes":
        hazards.append("fire")
    if smoke == "yes":
        hazards.append("smoke")
    if hazards:
        parts.append(f"They have perceived {' and '.join(hazards)}.")
    ev = list(events or [])
    if ev:
        # Events are newest-first in the Agent Info panel.
        recent = "; ".join(ev[:5])
        parts.append(f"Recent state changes: {recent}.")
    else:
        parts.append("No discrete state changes are recorded yet.")
    return " ".join(parts)


def summarize_or_fallback(raw: str, req: "SummarizeRequest") -> str:
    cleaned = clean_summary(raw)
    if cleaned:
        return cleaned
    return fallback_summary(req.agent_id, req.snapshot, req.events)
