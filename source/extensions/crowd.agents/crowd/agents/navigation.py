# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""NavMesh generation and path following (Phase 2 implementation target)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb

if TYPE_CHECKING:
    from .agent import Agent


class NavigationSystem:
    """Wraps Omniverse NavMesh APIs; stubs until Phase 2."""

    def generate_navmesh(self) -> bool:
        carb.log_info("[crowd.agents] NavigationSystem.generate_navmesh() — Phase 2 stub")
        return True

    def apply_intent(self, agent: "Agent", intent: str, dt: float) -> None:
        """Move agent toward goal based on intent. Placeholder for NavMesh follow."""
        # Phase 2: query path and step along it.
        # For now, leave pose unchanged unless a future waypoint fallback is added.
        _ = (agent, intent, dt)
