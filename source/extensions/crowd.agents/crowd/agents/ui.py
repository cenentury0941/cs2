# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Control panel for venue load, NavMesh, and multi-agent seating."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import omni.ui as ui

if TYPE_CHECKING:
    from .simulation import Simulation

WINDOW_TITLE = "Crowd Sim"
AGENT_INFO_TITLE = "Individual Agent Info"


class CrowdAgentsWindow:
    """Omni UI panel for Phase 3: Spawn zone + Seat_XXX agents."""

    def __init__(self, simulation: "Simulation") -> None:
        self._simulation = simulation
        self._status_label: Optional[ui.Label] = None
        self._agent_count_label: Optional[ui.Label] = None
        self._agent_count_model = ui.SimpleIntModel(8)
        self._agent_count_model.add_value_changed_fn(self._on_agent_count_changed)
        self._window = ui.Window(WINDOW_TITLE, width=400, height=560)
        self._build()

    def _build(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("SimulCrowd — Crowd Sim", height=0, style={"font_size": 16})
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
                    "Nav_XXX cubes → Include Volumes, then bake. Spawn → random Seat_001–100.",
                    height=0,
                    style={"color": 0xFFAAAAAA, "font_size": 12},
                )
                with ui.HStack(height=24, spacing=6):
                    ui.Label("Agents", width=56, height=0)
                    ui.IntSlider(self._agent_count_model, min=1, max=100)
                    self._agent_count_label = ui.Label("8", width=36, height=0)
                ui.Button(
                    "Create Nav Include Volumes (from Nav_XXX)",
                    clicked_fn=self._on_create_nav_volumes,
                    height=28,
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
        self._agent_count_label = None
        self._agent_count_model = None

    def _agent_count(self) -> int:
        try:
            return int(self._agent_count_model.get_value_as_int())
        except Exception:  # noqa: BLE001
            return 8

    def _on_agent_count_changed(self, model) -> None:
        if self._agent_count_label:
            try:
                self._agent_count_label.text = str(int(model.get_value_as_int()))
            except Exception:  # noqa: BLE001
                pass

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

    def _on_create_nav_volumes(self) -> None:
        self._simulation.create_nav_include_volumes()
        self.refresh_status()

    def _on_spawn_agents(self) -> None:
        self._simulation.spawn_agents(count=self._agent_count())
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


class AgentInfoWindow:
    """Placeholder panel for per-agent inspection (Phase 5)."""

    def __init__(self) -> None:
        self._window = ui.Window(AGENT_INFO_TITLE, width=320, height=600)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Individual Agent Info", height=0, style={"font_size": 16})
                ui.Spacer(height=8)
                ui.Label(
                    "Select an agent to inspect intent, goal, perception, and path.",
                    height=0,
                    word_wrap=True,
                    style={"color": 0xFFAAAAAA},
                )
                ui.Spacer(height=12)
                ui.Label("(Placeholder — coming in Phase 5)", height=0, style={"color": 0xFF888888})

    def destroy(self) -> None:
        self._window = None
