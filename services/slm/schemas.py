"""Request/response models for the frozen /v1/decide contract."""

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
