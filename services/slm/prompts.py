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
    "Do not invent information. Do not invent seats, exits, or destinations that are "
    "not present in PERSON or PERCEPTION. "
    "Return ONLY one valid JSON object with exactly these fields: intent, destination. "
    "Allowed intents are GO_TO_SEAT, EXIT, IDLE. "
    "destination must be a seat id, exit id, or null. "
    "Rules (apply in order): "
    "(1) if fire_visible or smoke_visible is true, intent must be EXIT and destination "
    "must be nearest_exit (or another exit from exits); "
    "(2) else if goal is an exit id (name starts with Exit), intent must be EXIT and "
    "destination must be that exit — never use GO_TO_SEAT with an exit destination; "
    "(3) else if at_destination is true, intent must be IDLE and destination must be null; "
    "(4) else if a seat is known (assigned_seat set, or goal is a seat id starting with Seat, "
    "or seat_known is true with a seat id available), intent must be GO_TO_SEAT and "
    "destination must be that seat id; "
    "(5) else intent must be IDLE and destination must be null. "
    "GO_TO_SEAT is only valid when destination is a seat id. "
    "EXIT is only valid when destination is an exit id."
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
