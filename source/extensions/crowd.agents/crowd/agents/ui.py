# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Control panel for venue load, NavMesh, and multi-agent seating."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import omni.ui as ui

if TYPE_CHECKING:
    from .simulation import Simulation

WINDOW_TITLE = "Crowd Agents"


class CrowdAgentsWindow:
    """Omni UI panel for Phase 3: Spawn zone + Seat.XXX agents."""

    def __init__(self, simulation: "Simulation") -> None:
        self._simulation = simulation
        self._status_label: Optional[ui.Label] = None
        self._window = ui.Window(WINDOW_TITLE, width=400, height=520)
        self._build()

    def _build(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("SimulCrowd — Crowd Agents", height=0, style={"font_size": 16})
                ui.Spacer(height=4)

                ui.Label("Venue", height=0, style={"font_size": 14})
                ui.Button("Open Venue…", clicked_fn=self._on_open_venue, height=28)
                ui.Spacer(height=6)

                ui.Label("USD Smoke Test", height=0, style={"font_size": 14})
                with ui.HStack(height=28, spacing=4):
                    ui.Button("Spawn Cube", clicked_fn=self._on_spawn_cube)
                    ui.Button("Move Cube +1m", clicked_fn=self._on_move_cube)
                ui.Spacer(height=6)

                ui.Label("Phase 3 — Seats", height=0, style={"font_size": 14})
                ui.Label(
                    "Spawn plane (random X/Y on NavMesh) → Seat.000–024 on Run",
                    height=0,
                    style={"color": 0xFFAAAAAA, "font_size": 12},
                )
                ui.Button("Generate NavMesh", clicked_fn=self._on_generate_navmesh, height=28)
                ui.Button(
                    "Spawn Agents (idle on Spawn)",
                    clicked_fn=self._on_spawn_agents,
                    height=28,
                )
                ui.Button("Re-assign Seats", clicked_fn=self._on_assign_seats, height=28)
                ui.Button(
                    "Set Random Destination (debug)",
                    clicked_fn=self._on_set_destination,
                    height=28,
                )
                with ui.HStack(height=28, spacing=4):
                    ui.Button("Run (to seats)", clicked_fn=self._on_run)
                    ui.Button("Pause", clicked_fn=self._on_pause)
                    ui.Button("Reset", clicked_fn=self._on_reset)
                ui.Button("Trigger Fire → Exits", clicked_fn=self._on_trigger_fire, height=28)
                ui.Spacer(height=8)

                ui.Label("Status", height=0, style={"font_size": 14})
                self._status_label = ui.Label(
                    self._simulation.status_text,
                    height=0,
                    word_wrap=True,
                    style={"color": 0xFFCCCCCC},
                )

    def refresh_status(self) -> None:
        if self._status_label:
            self._status_label.text = self._simulation.status_text

    def destroy(self) -> None:
        self._window = None
        self._status_label = None

    def _on_open_venue(self) -> None:
        self._simulation.open_venue_dialog()

    def _on_spawn_cube(self) -> None:
        self._simulation.spawn_test_cube()
        self.refresh_status()

    def _on_move_cube(self) -> None:
        self._simulation.move_test_cube(offset_m=(1.0, 0.0, 0.0))
        self.refresh_status()

    def _on_generate_navmesh(self) -> None:
        self._simulation.generate_navmesh()
        self.refresh_status()

    def _on_spawn_agents(self) -> None:
        self._simulation.spawn_agents()
        self.refresh_status()

    def _on_assign_seats(self) -> None:
        self._simulation.assign_seats()
        self.refresh_status()

    def _on_set_destination(self) -> None:
        self._simulation.set_random_destination()
        self.refresh_status()

    def _on_run(self) -> None:
        self._simulation.set_running(True)
        self.refresh_status()

    def _on_pause(self) -> None:
        self._simulation.set_running(False)
        self.refresh_status()

    def _on_reset(self) -> None:
        self._simulation.reset()
        self.refresh_status()

    def _on_trigger_fire(self) -> None:
        self._simulation.trigger_fire()
        self.refresh_status()
