# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Control panel for venue load, NavMesh, and multi-agent seating."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import omni.ui as ui

if TYPE_CHECKING:
    from .simulation import Simulation

WINDOW_TITLE = "Crowd Sim"
AGENT_INFO_TITLE = "Individual Agent Info"

_AGENT_INFO_FIELDS = (
    ("id", "ID"),
    ("intent", "Intent"),
    ("goal", "Goal"),
    ("assigned_seat", "Assigned seat"),
    ("exit_chosen", "Exit chosen"),
    ("fire_perceived", "Fire perceived"),
    ("smoke_perceived", "Smoke perceived"),
    ("at_destination", "At destination"),
    ("position", "Position (feet)"),
    ("destination", "Destination"),
    ("waypoints_left", "Waypoints left"),
)


class CrowdAgentsWindow:
    """Omni UI panel for Phase 3: Spawn zone + Seat_XXX agents."""

    def __init__(self, simulation: "Simulation") -> None:
        self._simulation = simulation
        self._status_label: Optional[ui.Label] = None
        self._agent_count_label: Optional[ui.Label] = None
        self._fire_button: Optional[ui.Button] = None
        self._slm_status_label: Optional[ui.Label] = None
        self._slm_url_field: Optional[ui.StringField] = None
        self._use_slm_model = ui.SimpleBoolModel(False)
        self._use_slm_model.add_value_changed_fn(self._on_use_slm_changed)
        self._agent_count_model = ui.SimpleIntModel(8)
        self._agent_count_model.add_value_changed_fn(self._on_agent_count_changed)
        self._window = ui.Window(WINDOW_TITLE, width=400, height=680)
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
                self._fire_button = ui.Button(
                    "Start Fire",
                    clicked_fn=self._on_fire_button,
                    height=28,
                )
                ui.Label(
                    "Start Fire → hover preview → click to place. "
                    "Agents evacuate only when fire is in FOV+range with clear LOS "
                    "(blue=seat, red=exit). Click an agent for Individual Agent Info.",
                    height=0,
                    word_wrap=True,
                    style={"color": 0xFFAAAAAA, "font_size": 12},
                )
                ui.Spacer(height=8)

                ui.Label("Phase 6 — Hosted SLM", height=0, style={"font_size": 14})
                ui.Label(
                    "Laptop never loads the model — batch POST /v1/decide ~1 Hz.",
                    height=0,
                    word_wrap=True,
                    style={"color": 0xFFAAAAAA, "font_size": 12},
                )
                with ui.HStack(height=22, spacing=6):
                    ui.CheckBox(self._use_slm_model, width=22)
                    ui.Label("Use hosted SLM", height=0)
                with ui.HStack(height=24, spacing=6):
                    ui.Label("URL", width=36, height=0)
                    self._slm_url_field = ui.StringField(height=22)
                    self._slm_url_field.model.set_value(self._simulation.slm_base_url)
                with ui.HStack(height=28, spacing=4):
                    ui.Button("Apply URL", clicked_fn=self._on_slm_apply_url)
                    ui.Button("Check Ready", clicked_fn=self._on_slm_check_ready)
                self._slm_status_label = ui.Label(
                    self._simulation.slm_status_text,
                    height=0,
                    word_wrap=True,
                    style={"color": 0xFFAAAAAA, "font_size": 12},
                )
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
        if self._slm_status_label:
            self._slm_status_label.text = self._simulation.slm_status_text
        if self._fire_button:
            self._fire_button.text = (
                "Cancel" if self._simulation.is_placing_fire else "Start Fire"
            )

    def destroy(self) -> None:
        self._window = None
        self._status_label = None
        self._agent_count_label = None
        self._fire_button = None
        self._slm_status_label = None
        self._slm_url_field = None
        self._use_slm_model = None
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

    def _on_use_slm_changed(self, model) -> None:
        try:
            enabled = bool(model.get_value_as_bool())
        except Exception:  # noqa: BLE001
            enabled = False
        self._simulation.set_use_slm(enabled)
        self.refresh_status()

    def _on_slm_apply_url(self) -> None:
        url = ""
        if self._slm_url_field is not None:
            try:
                url = self._slm_url_field.model.get_value_as_string()
            except Exception:  # noqa: BLE001
                url = ""
        self._simulation.set_slm_base_url(url)
        self.refresh_status()

    def _on_slm_check_ready(self) -> None:
        self._on_slm_apply_url()
        self._simulation.probe_slm_ready()
        self.refresh_status()

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

    def _on_fire_button(self) -> None:
        self._simulation.toggle_fire_placement()
        self.refresh_status()


class AgentInfoWindow:
    """Per-agent inspection panel driven by custom click-select (Phase 5)."""

    def __init__(self, simulation: "Simulation") -> None:
        self._simulation = simulation
        self._hint_label: Optional[ui.Label] = None
        self._value_labels: Dict[str, ui.Label] = {}
        self._log_label: Optional[ui.Label] = None
        self._window = ui.Window(AGENT_INFO_TITLE, width=320, height=600)
        self._build()
        self.refresh()

    def _build(self) -> None:
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Individual Agent Info", height=0, style={"font_size": 16})
                    ui.Spacer(height=4)
                    self._hint_label = ui.Label(
                        "Click an agent in the viewport to inspect.",
                        height=0,
                        word_wrap=True,
                        style={"color": 0xFFAAAAAA, "font_size": 12},
                    )
                    ui.Spacer(height=8)
                    for key, title in _AGENT_INFO_FIELDS:
                        with ui.HStack(height=0, spacing=6):
                            ui.Label(
                                title,
                                width=120,
                                height=0,
                                style={"color": 0xFFAAAAAA, "font_size": 12},
                            )
                            self._value_labels[key] = ui.Label(
                                "—",
                                height=0,
                                word_wrap=True,
                                style={"color": 0xFFEEEEEE, "font_size": 13},
                            )
                    ui.Spacer(height=10)
                    ui.Label("State changes", height=0, style={"font_size": 14})
                    ui.Label(
                        "Discrete events for this run (intent, goal, fire, arrival) — "
                        "newest first; cleared on Spawn/Reset.",
                        height=0,
                        word_wrap=True,
                        style={"color": 0xFF888888, "font_size": 11},
                    )
                    self._log_label = ui.Label(
                        "(no events yet)",
                        height=0,
                        word_wrap=True,
                        style={"color": 0xFFCCCCCC, "font_size": 12},
                    )

    def refresh(self) -> None:
        snap = self._simulation.selected_agent_snapshot()
        if snap is None:
            if self._hint_label:
                self._hint_label.text = (
                    "Click an agent in the viewport to inspect."
                    if self._simulation.has_agents
                    else "Spawn agents, then click one to inspect."
                )
            for label in self._value_labels.values():
                label.text = "—"
            self._refresh_log(agent_id=None)
            return

        if self._hint_label:
            self._hint_label.text = f"Inspecting {snap['id']} (not USD-selected)."
        for key, _title in _AGENT_INFO_FIELDS:
            label = self._value_labels.get(key)
            if label is not None:
                label.text = snap.get(key, "—")
        self._refresh_log(agent_id=snap["id"])

    def _refresh_log(self, agent_id: Optional[str]) -> None:
        if not self._log_label:
            return
        if agent_id:
            lines = self._simulation.state_event_lines_for_agent_or_sim(agent_id)
        else:
            # Full run history when nothing is selected
            lines = self._simulation.state_event_lines()
        if not lines:
            text = "(no events yet — run the sim / place fire)"
        else:
            text = "\n".join(lines)
        # Avoid rewriting identical text every frame (causes flicker / scroll jump)
        if self._log_label.text != text:
            self._log_label.text = text

    def destroy(self) -> None:
        self._window = None
        self._hint_label = None
        self._value_labels.clear()
        self._log_label = None
