# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Extension lifecycle: UI, simulation tick, and cleanup."""

from __future__ import annotations

import carb
import omni.ext
import omni.kit.app

from .simulation import Simulation
from .ui import CrowdAgentsWindow


class CrowdAgentsExtension(omni.ext.IExt):
    """Owns the Crowd Agents window and drives the simulation update loop."""

    def on_startup(self, _ext_id: str) -> None:
        carb.log_info("[crowd.agents] Extension startup")

        self._simulation = Simulation()
        self._window = CrowdAgentsWindow(self._simulation)
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="crowd.agents.update")
        )

    def _on_update(self, event) -> None:
        dt = event.payload.get("dt", 0.0) if event.payload else 0.0
        self._simulation.update(dt)
        self._window.refresh_status()

    def on_shutdown(self) -> None:
        carb.log_info("[crowd.agents] Extension shutdown")
        self._update_sub = None
        if self._window:
            self._window.destroy()
            self._window = None
        if self._simulation:
            self._simulation.shutdown()
            self._simulation = None
