# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Fire / smoke hazard state and perception helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb

if TYPE_CHECKING:
    from .agent import Agent


class FireSystem:
    """MVP fire flag + later VFX / volume perception."""

    def __init__(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def trigger(self) -> None:
        self._active = True
        carb.log_info("[crowd.agents] FireSystem.trigger() — fire active")

    def reset(self) -> None:
        self._active = False

    def is_visible_to(self, agent: "Agent") -> bool:
        """Phase 4: test agent against smoke/fire volumes. MVP: global when active."""
        _ = agent
        return self._active
