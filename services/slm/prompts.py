"""System and user prompt builders for SimulCrowd decisions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import AgentInput

SYSTEM_PROMPT = (
    "You are the decision engine for ONE simulated person in SimulCrowd. "
    "Your ONLY task is to select the person's next immediate action. "
    "The simulation engine handles navigation, pathfinding, movement, collision "
    "avoidance, physics, animation, and coordinates. You do NOT control these systems. "
    "Make a believable human decision using ONLY the information provided. "
    "Do not explain your decision. Do not provide reasoning. Do not output dialogue. "
    "Do not invent information. "
    "Return ONLY one valid JSON object with exactly these fields: intent, destination. "
    "Allowed intents are GO_TO_SEAT, EXIT, IDLE. "
    "destination must be a seat id, exit id, or null. "
    "Rules: if fire_visible or smoke_visible is true, prefer EXIT toward nearest_exit; "
    "if at_destination is true, prefer IDLE; otherwise prefer GO_TO_SEAT toward "
    "assigned_seat or goal when a seat is known."
)


def build_user_prompt(agent: "AgentInput") -> str:
    """Build the user message from one agent's batch fields."""
    person = {
        "id": agent.id,
        "goal": agent.goal,
        "assigned_seat": agent.assigned_seat,
        "current_intent": agent.intent,
    }
    perception = agent.perception or {}
    return (
        f"PERSON: {json.dumps(person, separators=(',', ':'))}\n\n"
        f"PERCEPTION: {json.dumps(perception, separators=(',', ':'))}\n\n"
        "Select the next immediate intent."
    )
