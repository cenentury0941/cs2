"""Shared agent-state scenarios for SLM behavior checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class Scenario(TypedDict, total=False):
    name: str
    agent: Dict[str, Any]
    expect_intent: str
    expect_destination: Optional[str]
    # If True, destination must be null/None
    expect_destination_null: bool
    notes: str


def _agent(
    id: str,
    *,
    goal: Optional[str] = None,
    assigned_seat: Optional[str] = None,
    intent: str = "WALK",
    fire_visible: bool = False,
    smoke_visible: bool = False,
    seat_known: bool = True,
    at_destination: bool = False,
    nearest_exit: str = "Exit_A",
    exits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": id,
        "goal": goal,
        "assigned_seat": assigned_seat,
        "intent": intent,
        "perception": {
            "fire_visible": fire_visible,
            "smoke_visible": smoke_visible,
            "seat_known": seat_known,
            "at_destination": at_destination,
            "nearest_exit": nearest_exit,
            "exits": exits or ["Exit_A", "Exit_B"],
        },
    }


# Expected intents follow SYSTEM_PROMPT + fallback_fields priority:
# fire/smoke → EXIT; at_destination → IDLE; else seat → GO_TO_SEAT; else exit goal → EXIT; else IDLE
SCENARIOS: List[Scenario] = [
    {
        "name": "walk_to_assigned_seat",
        "agent": _agent(
            "WalkSeat",
            goal="Seat_003",
            assigned_seat="Seat_003",
            seat_known=True,
            at_destination=False,
        ),
        "expect_intent": "GO_TO_SEAT",
        "expect_destination": "Seat_003",
        "notes": "Normal seating path",
    },
    {
        "name": "seat_known_goal_only",
        "agent": _agent(
            "GoalSeat",
            goal="Seat_012",
            assigned_seat=None,
            seat_known=True,
            at_destination=False,
        ),
        "expect_intent": "GO_TO_SEAT",
        "expect_destination": "Seat_012",
        "notes": "Goal is a seat when assigned_seat missing",
    },
    {
        "name": "arrived_at_seat",
        "agent": _agent(
            "Seated",
            goal="Seat_005",
            assigned_seat="Seat_005",
            at_destination=True,
        ),
        "expect_intent": "IDLE",
        "expect_destination_null": True,
        "notes": "At destination without hazard → idle",
    },
    {
        "name": "fire_visible_evacuate",
        "agent": _agent(
            "SeeFire",
            goal="Seat_003",
            assigned_seat="Seat_003",
            fire_visible=True,
            nearest_exit="Exit_A",
        ),
        "expect_intent": "EXIT",
        "expect_destination": "Exit_A",
        "notes": "Fire overrides seating",
    },
    {
        "name": "smoke_visible_evacuate",
        "agent": _agent(
            "SeeSmoke",
            goal="Seat_003",
            assigned_seat="Seat_003",
            smoke_visible=True,
            nearest_exit="Exit_B",
        ),
        "expect_intent": "EXIT",
        "expect_destination": "Exit_B",
        "notes": "Smoke alone should evacuate toward nearest_exit",
    },
    {
        "name": "fire_while_seated",
        "agent": _agent(
            "FireSeated",
            goal="Seat_001",
            assigned_seat="Seat_001",
            fire_visible=True,
            at_destination=True,
            nearest_exit="Exit_B",
        ),
        "expect_intent": "EXIT",
        "expect_destination": "Exit_B",
        "notes": "Hazard beats at_destination / IDLE",
    },
    {
        "name": "already_heading_to_exit",
        "agent": _agent(
            "ToExit",
            goal="Exit_A",
            assigned_seat=None,
            seat_known=False,
            at_destination=False,
            fire_visible=False,
        ),
        "expect_intent": "EXIT",
        "expect_destination": "Exit_A",
        "notes": "Goal is an exit id with no assigned seat",
    },
    {
        "name": "at_exit_idle",
        "agent": _agent(
            "AtExit",
            goal="Exit_A",
            assigned_seat="Seat_002",
            at_destination=True,
            fire_visible=False,
            smoke_visible=False,
        ),
        "expect_intent": "IDLE",
        "expect_destination_null": True,
        "notes": "Reached exit / destination with no hazard",
    },
    {
        "name": "unknown_seat_no_hazard",
        "agent": _agent(
            "Lost",
            goal=None,
            assigned_seat=None,
            seat_known=False,
            at_destination=False,
        ),
        "expect_intent": "IDLE",
        "expect_destination_null": True,
        "notes": "No seat/exit goal and no hazard",
    },
    {
        "name": "fire_prefers_nearest_exit_b",
        "agent": _agent(
            "NearB",
            goal="Seat_009",
            assigned_seat="Seat_009",
            fire_visible=True,
            smoke_visible=True,
            nearest_exit="Exit_B",
            exits=["Exit_A", "Exit_B"],
        ),
        "expect_intent": "EXIT",
        "expect_destination": "Exit_B",
        "notes": "Use perception.nearest_exit",
    },
    {
        "name": "walking_intent_still_redecide",
        "agent": _agent(
            "WasWalking",
            goal="Seat_007",
            assigned_seat="Seat_007",
            intent="WALK",
            at_destination=False,
        ),
        "expect_intent": "GO_TO_SEAT",
        "expect_destination": "Seat_007",
        "notes": "Prior WALK intent should not block seating decision",
    },
    {
        "name": "idle_intent_but_fire",
        "agent": _agent(
            "IdleThenFire",
            goal="Seat_004",
            assigned_seat="Seat_004",
            intent="IDLE",
            fire_visible=True,
            at_destination=True,
            nearest_exit="Exit_A",
        ),
        "expect_intent": "EXIT",
        "expect_destination": "Exit_A",
        "notes": "Current IDLE must yield to fire",
    },
]


def scenarios_as_agents() -> List[Dict[str, Any]]:
    return [dict(s["agent"]) for s in SCENARIOS]
