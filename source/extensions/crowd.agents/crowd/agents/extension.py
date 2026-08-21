# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Extension lifecycle: UI, simulation tick, and cleanup."""

from __future__ import annotations

import carb
import omni.ext
import omni.kit.app

from .layout import schedule_startup_layout
from .simulation import Simulation
from .ui import AgentInfoWindow, CrowdAgentsWindow

# Match crowd.sim.kit rate limit; used if the update event omits dt.
_DEFAULT_DT = 1.0 / 30.0


class CrowdAgentsExtension(omni.ext.IExt):
    """Owns Crowd Sim UI and drives the simulation update loop."""

    def on_startup(self, _ext_id: str) -> None:
        carb.log_info("[crowd.agents] Extension startup")

        self._simulation = Simulation()
        self._window = CrowdAgentsWindow(self._simulation)
        self._agent_info_window = AgentInfoWindow()
        self._layout_task = schedule_startup_layout()
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="crowd.agents.update")
        )

    def _on_update(self, event) -> None:
        dt = self._read_dt(event)
        self._simulation.update(dt)
        self._window.refresh_status()

    @staticmethod
    def _read_dt(event) -> float:
        """Kit update payloads vary; never allow a zero dt while simulating."""
        try:
            payload = event.payload
            if payload is None:
                return _DEFAULT_DT
            # carb.dictionary often supports [] but not dict.get reliably
            try:
                value = payload["dt"]
            except Exception:  # noqa: BLE001
                value = payload.get("dt") if hasattr(payload, "get") else None
            if value is None:
                return _DEFAULT_DT
            dt = float(value)
            return dt if dt > 0.0 else _DEFAULT_DT
        except Exception:  # noqa: BLE001
            return _DEFAULT_DT

    def on_shutdown(self) -> None:
        carb.log_info("[crowd.agents] Extension shutdown")
        self._update_sub = None
        self._layout_task = None
        if self._agent_info_window:
            self._agent_info_window.destroy()
            self._agent_info_window = None
        if self._window:
            self._window.destroy()
            self._window = None
        if self._simulation:
            self._simulation.shutdown()
            self._simulation = None
