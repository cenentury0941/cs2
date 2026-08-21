# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Pluggable decision engine: rule-based now, SLM later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


class DecisionEngine(ABC):
    """High-level intent only — navigation owns movement."""

    @abstractmethod
    def decide(self, agent: "Agent") -> str:
        """Return an intent string such as GO_TO_SEAT, EXIT, or IDLE."""


class RuleBasedDecisionEngine(DecisionEngine):
    """Hackathon-ready brain; swap for SLMDecisionEngine without changing the loop."""

    def decide(self, agent: "Agent") -> str:
        perception = agent.perception or {}
        if perception.get("fire_visible"):
            return "EXIT"
        if perception.get("at_destination"):
            return "IDLE"
        goal = agent.goal or ""
        if goal.startswith("Exit"):
            return "EXIT"
        if agent.destination is not None or agent.assigned_seat or goal.startswith("Seat"):
            return "GO_TO_SEAT"
        if agent.destination is not None or goal:
            return "GO_TO_SEAT"
        return "IDLE"

