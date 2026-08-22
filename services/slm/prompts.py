"""System and user prompt builders for SimulCrowd decisions and summaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas import AgentInput, SummarizeRequest

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

SUMMARIZE_SYSTEM_PROMPT = (
    "You write a short plain-language status summary for ONE simulated person in "
    "SimulCrowd, for an operator viewing the Individual Agent Info panel. "
    "Use ONLY the SNAPSHOT and EVENTS provided. Do not invent fire, seats, exits, "
    "intents, or outcomes that are not present. Do not output JSON, bullet lists, "
    "headings, or dialogue. Write 2–4 concise sentences covering: who they are "
    "(id), what they are doing now (intent/goal/destination), notable hazards "
    "they have perceived, and the key state changes from the event log "
    "(chronological sense, even if events are listed newest-first)."
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


def build_summarize_user_prompt(req: "SummarizeRequest") -> str:
    """Build the user message from Agent Info snapshot + state-change lines."""
    snapshot = dict(req.snapshot or {})
    if "id" not in snapshot:
        snapshot["id"] = req.agent_id
    events = list(req.events or [])
    events_block = "\n".join(events) if events else "(no events yet)"
    return (
        f"AGENT_ID: {req.agent_id}\n\n"
        f"SNAPSHOT: {json.dumps(snapshot, separators=(',', ':'))}\n\n"
        f"EVENTS (newest first):\n{events_block}\n\n"
        "Write the operator summary now."
    )
