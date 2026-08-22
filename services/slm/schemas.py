"""Request/response models for /v1/decide and /v1/summarize."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["GO_TO_SEAT", "EXIT", "IDLE"]


class AgentInput(BaseModel):
    id: str
    goal: Optional[str] = None
    assigned_seat: Optional[str] = None
    intent: Optional[str] = None
    perception: Dict[str, Any] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    request_id: str
    agents: List[AgentInput]


class Decision(BaseModel):
    id: str
    intent: Intent
    destination: Optional[str] = None


class DecideResponse(BaseModel):
    request_id: str
    decisions: List[Decision]


class SummarizeRequest(BaseModel):
    """Payload mirroring Individual Agent Info (snapshot + state-change log)."""

    request_id: str
    agent_id: str
    # Flat string fields from selected_agent_snapshot() when available.
    snapshot: Dict[str, str] = Field(default_factory=dict)
    # Newest-first display lines from state_event_lines_for_agent_or_sim().
    events: List[str] = Field(default_factory=list)


class SummarizeResponse(BaseModel):
    request_id: str
    agent_id: str
    summary: str
