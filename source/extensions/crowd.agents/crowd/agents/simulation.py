# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Simulation orchestrator: agents, navigation, fire, and decision engine."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, UsdGeom

from .agent import Agent
from .ai import DecisionEngine, RuleBasedDecisionEngine
from .fire import FIRE_FOV_DEG, FIRE_VIEW_RANGE_M, FireSystem
from .inspect import AgentInspectPicker, pick_frontmost_along_ray
from .navigation import NavigationSystem
from . import scene as scene_util
from .viz import clear_fov_wireframe, sync_fov_wireframe

TEST_CUBE_PATH = "/World/CrowdAgents/TestCube"
AGENTS_ROOT = "/World/CrowdAgents"
DEFAULT_AGENT_COUNT = 8
MAX_AGENTS = 100
MIN_AGENTS = 1
AGENT_HEIGHT = 1.7
AGENT_RADIUS = 0.35
# Soft safety cap only — events persist for the whole run until spawn/reset.
MAX_STATE_EVENTS = 10000
Vec3 = Tuple[float, float, float]


def _is_agent_prim_path(path: str) -> bool:
    """True for /World/CrowdAgents/Agent_XXX (and descendants)."""
    p = (path or "").replace("\\", "/")
    prefix = f"{AGENTS_ROOT}/Agent_"
    if not p.startswith(prefix):
        return False
    rest = p[len(AGENTS_ROOT) + 1 :]  # Agent_001[/…]
    name = rest.split("/", 1)[0]
    return name.startswith("Agent_")


class Simulation:
    """Owns agents and wires decision → navigation → USD each tick."""

    def __init__(self) -> None:
        self._running = False
        self._agents: List[Agent] = []
        self._navigation = NavigationSystem()
        self._fire = FireSystem()
        self._brain: DecisionEngine = RuleBasedDecisionEngine()
        self._status = (
            "Ready — bake NavMesh, Spawn Agents (uses Spawn + Seat_XXX), Run."
        )
        self._log_lines: List[str] = []
        self._exits = []
        self._rng = random.Random(7)
        self._selected_agent_id: Optional[str] = None
        self._hovered_agent_id: Optional[str] = None
        self._suppress_selection_event = False
        self._inspect_picker = AgentInspectPicker(self)
        self._sim_time = 0.0
        self._state_events: List[Dict[str, str]] = []
        self._agent_watch: Dict[str, Dict[str, object]] = {}
        self._stage_event_sub = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop(
                self._on_stage_event, name="crowd.agents.agent_inspect"
            )
        )

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def selected_agent_id(self) -> Optional[str]:
        return self._selected_agent_id

    def get_selected_agent(self) -> Optional[Agent]:
        if not self._selected_agent_id:
            return None
        for agent in self._agents:
            if agent.agent_id == self._selected_agent_id:
                return agent
        return None

    def selected_agent_snapshot(self) -> Optional[Dict[str, str]]:
        """Flat string fields for the Individual Agent Info panel."""
        agent = self.get_selected_agent()
        if agent is None:
            return None
        goal = agent.goal or "—"
        exit_chosen = goal if goal.startswith("Exit") else "—"
        fire_seen = bool(agent.memory.get("fire_seen") or agent.perception.get("fire_visible"))
        pos = agent.position
        dest = agent.destination
        remaining_wp = max(0, len(agent.waypoints) - agent.waypoint_index)
        return {
            "id": agent.agent_id,
            "intent": agent.intent or "—",
            "goal": goal,
            "assigned_seat": agent.assigned_seat or "—",
            "exit_chosen": exit_chosen,
            "fire_perceived": "yes" if fire_seen else "no",
            "smoke_perceived": (
                "yes" if agent.perception.get("smoke_visible") else "no"
            ),
            "at_destination": (
                "yes" if agent.perception.get("at_destination") else "no"
            ),
            "position": f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
            "destination": (
                f"({dest[0]:.2f}, {dest[1]:.2f}, {dest[2]:.2f})"
                if dest is not None
                else "—"
            ),
            "waypoints_left": str(remaining_wp),
        }

    def state_event_lines(
        self, *, agent_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """Newest-first discrete state-change lines for the current run.

        Cleared only on spawn / reset / new venue — not trimmed during a run
        (aside from a large soft safety cap).
        """
        events = self._state_events
        if agent_id:
            events = [e for e in events if e.get("agent_id") == agent_id]
        lines = [e["line"] for e in reversed(events)]
        if limit is not None and limit > 0:
            return lines[:limit]
        return lines

    def state_event_lines_for_agent_or_sim(
        self, agent_id: str, *, limit: Optional[int] = None
    ) -> List[str]:
        """Newest-first events for one agent plus sim-level (fire/run/pause)."""
        events = [
            e
            for e in self._state_events
            if e.get("agent_id") in (agent_id, "sim")
        ]
        lines = [e["line"] for e in reversed(events)]
        if limit is not None and limit > 0:
            return lines[:limit]
        return lines

    @property
    def has_agents(self) -> bool:
        return bool(self._agents)

    def select_agent(self, agent_id: Optional[str]) -> None:
        """Set inspected agent for the info panel (not USD selection)."""
        if agent_id is None:
            self._selected_agent_id = None
            self.refresh_visuals()
            return
        if any(a.agent_id == agent_id for a in self._agents):
            self._selected_agent_id = agent_id
            self._clear_agent_usd_selection()
            self.refresh_visuals()

    def set_hovered_agent(self, agent_id: Optional[str]) -> None:
        if agent_id == self._hovered_agent_id:
            return
        if agent_id is not None and not any(a.agent_id == agent_id for a in self._agents):
            agent_id = None
        self._hovered_agent_id = agent_id
        self.refresh_visuals()

    def pick_frontmost_agent(
        self, origin: Vec3, direction: Vec3
    ) -> Optional[Agent]:
        """Frontmost agent capsule along a camera ray (correct depth order)."""
        return pick_frontmost_along_ray(
            self._agents, origin, direction, up=scene_util.up_axis()
        )

    def refresh_visuals(self) -> None:
        """Apply highlight colors + selected FOV wireframe (works while paused)."""
        selected = self._selected_agent_id
        hovered = self._hovered_agent_id
        for agent in self._agents:
            if agent.agent_id == selected:
                highlight = "selected"
            elif agent.agent_id == hovered:
                highlight = "hovered"
            else:
                highlight = None
            agent.sync_to_usd(highlight=highlight)

        sel = self.get_selected_agent()
        if sel is None:
            clear_fov_wireframe()
        else:
            sync_fov_wireframe(
                sel.position,
                sel.facing,
                up=scene_util.up_axis(),
                height=sel.height,
            )

    def shutdown(self) -> None:
        self._stage_event_sub = None
        if self._inspect_picker:
            self._inspect_picker.destroy()
            self._inspect_picker = None
        clear_fov_wireframe()
        self._fire.reset()
        self._agents.clear()
        self._selected_agent_id = None
        self._hovered_agent_id = None
        self._agent_watch.clear()
        self._state_events.clear()
        self._sim_time = 0.0
        self._running = False

    def _on_stage_event(self, event) -> None:
        try:
            if event.type != int(omni.usd.StageEventType.SELECTION_CHANGED):
                return
        except Exception:  # noqa: BLE001
            return
        if self._suppress_selection_event:
            return
        self._adopt_agent_clicks_from_usd_selection()

    def _adopt_agent_clicks_from_usd_selection(self) -> None:
        """Clicking an agent inspects it in the panel; clear traditional selection."""
        selection = omni.usd.get_context().get_selection()
        try:
            paths = list(selection.get_selected_prim_paths() or [])
        except Exception:  # noqa: BLE001
            return
        if not paths:
            return

        chosen: Optional[Agent] = None
        agent_paths = []
        other_paths = []
        for path in paths:
            if not _is_agent_prim_path(path):
                other_paths.append(path)
                continue
            agent_paths.append(path)
            if chosen is None:
                chosen = self._find_agent_by_prim_path(path)

        if not agent_paths:
            return

        if chosen is not None:
            self._selected_agent_id = chosen.agent_id

        self._clear_agent_usd_selection(keep_other_paths=other_paths)
        self.refresh_visuals()

    def _clear_agent_usd_selection(
        self, keep_other_paths: Optional[List[str]] = None
    ) -> None:
        """Drop agent prims from USD selection so Stage/Property never keep them."""
        ctx = omni.usd.get_context()
        selection = ctx.get_selection()
        self._suppress_selection_event = True
        try:
            if keep_other_paths is not None:
                if keep_other_paths:
                    selection.set_selected_prim_paths(keep_other_paths)
                else:
                    selection.clear_selected_prim_paths()
                return
            paths = list(selection.get_selected_prim_paths() or [])
            remaining = [p for p in paths if not _is_agent_prim_path(p)]
            if len(remaining) == len(paths):
                return
            if remaining:
                selection.set_selected_prim_paths(remaining)
            else:
                selection.clear_selected_prim_paths()
        except Exception:  # noqa: BLE001
            try:
                selection.clear_selected_prim_paths()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._suppress_selection_event = False

    def _find_agent_by_prim_path(self, path: str) -> Optional[Agent]:
        p = (path or "").replace("\\", "/")
        for agent in self._agents:
            ap = agent.prim_path.replace("\\", "/")
            if p == ap or p.startswith(ap + "/"):
                return agent
        parts = p.split("/")
        for part in parts:
            if part.startswith("Agent_"):
                for agent in self._agents:
                    if agent.agent_id == part:
                        return agent
        return None

    def _set_agents_pickable(self, pickable: bool) -> None:
        ctx = omni.usd.get_context()
        for agent in self._agents:
            try:
                ctx.set_pickable(agent.prim_path, pickable)
            except Exception:  # noqa: BLE001
                pass

    def update(self, dt: float) -> None:
        if self._inspect_picker:
            self._inspect_picker.ensure_attached()
        if not self._running or dt <= 0.0:
            return

        self._sim_time += dt
        fire_active = self._fire.is_active
        seated = 0
        evacuating = 0
        saw_fire = 0
        up = scene_util.up_axis()
        self._fire.tick(dt)

        for agent in self._agents:
            visible = self._fire.is_visible_to(agent, up=up)
            agent.perception["fire_visible"] = visible
            agent.perception["smoke_visible"] = visible
            if visible:
                saw_fire += 1

            brain_intent = self._brain.decide(agent)
            intent = brain_intent

            if intent == "EXIT":
                # Leave seat / prior arrival so path following can start
                if agent.perception.get("at_destination") and not (
                    agent.goal or ""
                ).startswith("Exit"):
                    agent.perception["at_destination"] = False
                    agent.waypoints = []
                    agent.waypoint_index = 0
                self._ensure_exit_destination(agent)
                if agent.destination is not None and not agent.perception.get("at_destination"):
                    if not agent.waypoints:
                        self._navigation.set_destination(agent, agent.destination)
                    intent = "WALK"
            elif (
                intent == "GO_TO_SEAT"
                and agent.destination is not None
                and not agent.perception.get("at_destination")
            ):
                intent = "WALK"

            agent.intent = intent
            self._navigation.apply_intent(agent, intent, dt)
            self._track_agent_state_changes(agent, brain_intent)

        # Only moving agents collide; seated agents are ignored by others
        self._separate_walking_agents(up)

        self.refresh_visuals()

        for agent in self._agents:
            if agent.perception.get("at_destination"):
                if (agent.goal or "").startswith("Exit"):
                    evacuating += 1
                else:
                    seated += 1

        total = len(self._agents)
        if fire_active:
            fleeing = sum(
                1 for a in self._agents if (a.goal or "").startswith("Exit")
            )
            self._status = (
                f"Running (FIRE) — {saw_fire}/{total} see fire, "
                f"{fleeing} evacuating, {evacuating} at exits"
            )
        else:
            walking = total - seated
            self._status = f"Running — {seated}/{total} seated, {walking} walking"

    def set_running(self, running: bool) -> None:
        if running and not self._agents:
            self._status = "Spawn Agents first (needs Spawn plane + Seat_XXX)."
            self._log(self._status)
            return
        if running:
            self._exits = scene_util.find_exits()
            for agent in self._agents:
                # Already evacuating (saw fire earlier) — repath to exit
                if self._fire.is_active and (agent.goal or "").startswith("Exit"):
                    self._ensure_exit_destination(agent)
                    if agent.destination is not None:
                        agent.perception["at_destination"] = False
                        self._navigation.set_destination(agent, agent.destination)
                elif agent.destination is not None:
                    # Prompted walk: Spawn → assigned Seat_XXX
                    agent.perception["at_destination"] = False
                    self._navigation.set_destination(agent, agent.destination)
            self.refresh_visuals()
            self._running = True
            self._note_state_change("sim", "run")
            self._status = f"Running — {len(self._agents)} agents walking Spawn → seats"
            self._log(self._status)
            return

        self._running = False
        self._note_state_change("sim", "paused")
        self._status = "Paused"
        self._log(self._status)

    def reset(self) -> None:
        self._running = False
        self._fire.reset()
        for agent in self._agents:
            agent.reset()
            agent.waypoints = []
            agent.waypoint_index = 0
            agent.intent = "IDLE"
            agent.perception["at_destination"] = False
            self._seed_agent_watch(agent, "IDLE")
            self._note_state_change(agent.agent_id, "reset")
        self.refresh_visuals()
        self._status = (
            f"Reset — {len(self._agents)} agents back at Spawn (idle). Press Run to seats."
        )
        self._log("Simulation reset")

    def open_venue_dialog(self) -> None:
        """Open a USD/GLB via the Kit file importer when available."""
        try:
            from omni.kit.window.file_importer import get_file_importer
        except ImportError:
            self._status = "File importer unavailable — use File > Open."
            carb.log_warn(self._status)
            return

        importer = get_file_importer()
        if importer is None:
            self._status = "File importer unavailable — use File > Open."
            return

        def _on_import(filename: str, dirname: str, selections: list = None, **_kwargs) -> None:
            if not filename:
                return
            path = f"{dirname}/{filename}".replace("\\", "/")
            if dirname.endswith("/") or dirname.endswith("\\"):
                path = f"{dirname}{filename}".replace("\\", "/")
            ok = omni.usd.get_context().open_stage(path)
            if ok:
                self._agents.clear()
                self._selected_agent_id = None
                self._hovered_agent_id = None
                self._agent_watch.clear()
                self._state_events.clear()
                self._sim_time = 0.0
                clear_fov_wireframe()
                self._status = f"Opened: {filename}"
                self._log(self._status)
            else:
                self._status = f"Failed to open: {path}"
                carb.log_error(self._status)

        importer.show_window(
            title="Open Venue",
            import_button_label="Open",
            import_handler=_on_import,
            file_extension_types=[
                (".usd", "USD"),
                (".usda", "USD ASCII"),
                (".usdc", "USD Crate"),
                (".usdz", "USDZ"),
                (".glb", "glTF Binary"),
                (".gltf", "glTF"),
            ],
        )
        self._status = "Select a venue file…"

    def spawn_test_cube(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status = "No stage — open or create a stage first."
            carb.log_warn(self._status)
            return

        self._ensure_xform(stage, "/World")
        self._ensure_xform(stage, AGENTS_ROOT)
        if stage.GetPrimAtPath(TEST_CUBE_PATH):
            stage.RemovePrim(TEST_CUBE_PATH)

        cube = UsdGeom.Cube.Define(stage, TEST_CUBE_PATH)
        cube.CreateSizeAttr(1.0)
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self._lift_point((0.0, 0.0, 0.0), 0.5)))

        self._status = f"Spawned cube at {TEST_CUBE_PATH}"
        self._log(self._status)

    def move_test_cube(self, offset_m: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status = "No stage open."
            return

        prim = stage.GetPrimAtPath(TEST_CUBE_PATH)
        if not prim or not prim.IsValid():
            self._status = "No test cube — click Spawn Cube first."
            return

        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.5))

        current = translate_op.Get()
        if current is None:
            current = Gf.Vec3d(0.0, 0.0, 0.5)

        new_pos = Gf.Vec3d(
            current[0] + offset_m[0],
            current[1] + offset_m[1],
            current[2] + offset_m[2],
        )
        translate_op.Set(new_pos)
        self._status = f"Moved cube to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
        self._log(self._status)

    def generate_navmesh(self) -> None:
        ok = self._navigation.generate_navmesh()
        if ok:
            info = scene_util.describe_scene()
            self._status = (
                f"NavMesh OK — Spawn={info['spawn']}, "
                f"seats={info['seat_count']}, exits={info['exits']}"
            )
        else:
            self._status = (
                "NavMesh bake failed — create/scale Include Volume over the floor, "
                "then try again (or bake from Window > Navigation > NavMesh)."
            )
        self._log(self._status)

    def create_nav_include_volumes(self) -> None:
        """Convert Nav_XXX placeholder cubes into NavMesh Include Volumes."""
        count, message = self._navigation.create_include_volumes_from_nav_markers()
        self._status = message
        self._log(self._status)
        if count > 0:
            carb.log_info(f"[crowd.agents] {message}")

    def spawn_agents(self, count: Optional[int] = None) -> None:
        """Phase 3: spawn N agents inside Spawn bounds, each assigned a Seat_XXX."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status = "No stage open."
            return

        if not self._navigation.has_navmesh():
            if not self._navigation.generate_navmesh() and not self._navigation.has_navmesh():
                self._status = "No NavMesh — bake first (Generate NavMesh)."
                self._log(self._status)
                return

        spawn_zone = scene_util.find_spawn_zone(stage)
        if spawn_zone is None:
            self._status = 'No prim named "Spawn" found — add a Spawn plane for the spawn zone.'
            self._log(self._status)
            return

        seats = scene_util.find_seats(stage)
        if not seats:
            self._status = (
                "No Seat_XXX prims found (expected Seat_001 … Seat_100)."
            )
            self._log(self._status)
            return

        self._exits = scene_util.find_exits(stage)
        wanted = count if count is not None else DEFAULT_AGENT_COUNT
        wanted = max(MIN_AGENTS, min(int(wanted), MAX_AGENTS, len(seats)))

        self._clear_agent_prims(stage)
        self._ensure_xform(stage, "/World")
        self._ensure_xform(stage, AGENTS_ROOT)

        agents: List[Agent] = []
        assigned_seats = self._sample_seats(seats, wanted)
        up = scene_util.up_axis(stage)
        placed: List[Vec3] = []
        # Tighter spawn spacing only when packing many agents
        min_spacing = 0.95 if wanted <= 20 else (0.65 if wanted <= 50 else 0.45)

        if spawn_zone.area < 0.05:
            self._status = (
                f"Spawn plane area too small ({spawn_zone.area:.3f}). "
                f"geom={spawn_zone.geom_path} — Apply Scale so X/Y extents are real meters."
            )
            self._log(self._status)
            return

        for i, seat in enumerate(assigned_seats):
            feet = scene_util.sample_spawn_feet_on_navmesh(
                spawn_zone,
                snap_fn=self._navigation.snap_to_navmesh,
                rng=self._rng,
                existing=placed,
                min_spacing=min_spacing,
            )
            if feet is None:
                self._status = (
                    f"Could only place {len(agents)}/{wanted} agents on Spawn plane + NavMesh "
                    f"(area={spawn_zone.area:.1f}, geom={spawn_zone.geom_path})."
                )
                self._log(self._status)
                break

            placed.append(feet)
            prim_path = f"{AGENTS_ROOT}/Agent_{i + 1:03d}"
            self._create_capsule(stage, prim_path, up)

            seat_pos = self._navigation.snap_to_navmesh(seat.position) or seat.position
            agent = Agent(
                agent_id=f"Agent_{i + 1:03d}",
                prim_path=prim_path,
                position=feet,
                goal=seat.name,
                destination=seat_pos,
                assigned_seat=seat.name,
                seat_prim_path=seat.prim_path,
                height=AGENT_HEIGHT,
                radius=AGENT_RADIUS,
                lane_bias=self._rng.triangular(-1.0, 1.0, 0.0),
            )
            agent.spawn_position = feet
            agent.memory["seat_position"] = list(seat_pos)
            agent.memory["lane_bias"] = agent.lane_bias
            agent.perception["seat_known"] = True
            agent.perception["at_destination"] = False
            agent.waypoints = []
            agent.waypoint_index = 0
            agent.intent = "IDLE"
            agent.face_toward(seat_pos, up)
            agents.append(agent)

        self._agents = agents
        self._selected_agent_id = None
        self._hovered_agent_id = None
        self._agent_watch.clear()
        self._state_events.clear()
        self._sim_time = 0.0
        clear_fov_wireframe()
        if not agents:
            self._status = (
                "Failed to spawn on Spawn plane — ensure it overlaps NavMesh and has real X/Y size."
            )
            self._log(self._status)
            return

        for agent in agents:
            self._seed_agent_watch(agent, "IDLE")
            seat = agent.assigned_seat or "—"
            self._note_state_change(agent.agent_id, "spawned", f"seat {seat}")

        self._set_agents_pickable(False)
        if self._inspect_picker:
            self._inspect_picker.ensure_attached()
        self.refresh_visuals()

        exit_names = [e.name for e in self._exits] or ["(none)"]
        seat_names = [s.name for s in assigned_seats[: len(agents)]]
        self._status = (
            f"Spawned {len(agents)} idle on Spawn (area={spawn_zone.area:.1f}) → "
            f"random seats {seat_names[0]}…{seat_names[-1]} "
            f"({len(seat_names)} of {len(seats)} available). "
            f"Exits: {', '.join(exit_names)}. Click an agent for Individual Agent Info."
        )
        self._log(self._status)

    def set_random_destination(self) -> None:
        """Debug: random NavMesh goals (Phase 2). Prefer seats via Spawn Agents."""
        if not self._agents:
            self._status = "Spawn Agents first."
            return

        planned = 0
        for agent in self._agents:
            dest = self._navigation.random_point(seed_key="crowd.dest")
            if dest is None:
                continue
            if self._navigation.set_destination(agent, dest):
                agent.goal = "random"
                agent.assigned_seat = None
                planned += 1

        self.refresh_visuals()
        self._status = f"Random destinations set for {planned}/{len(self._agents)} agents."
        self._log(self._status)

    def assign_seats(self) -> None:
        """Re-bind current agents to random unique Seat_XXX (paths planned on Run)."""
        if not self._agents:
            self._status = "Spawn Agents first."
            return
        seats = scene_util.find_seats()
        if not seats:
            self._status = "No Seat_XXX prims found (expected Seat_001 … Seat_100)."
            return

        assigned = self._sample_seats(seats, len(self._agents))
        for agent, seat in zip(self._agents, assigned):
            seat_pos = self._navigation.snap_to_navmesh(seat.position) or seat.position
            agent.assigned_seat = seat.name
            agent.seat_prim_path = seat.prim_path
            agent.goal = seat.name
            agent.destination = seat_pos
            agent.memory["seat_position"] = list(seat_pos)
            agent.perception["seat_known"] = True
            agent.perception["at_destination"] = False
            agent.waypoints = []
            agent.waypoint_index = 0
            agent.intent = "IDLE"
            agent.face_toward(seat_pos)
            self._track_agent_state_changes(agent, "IDLE")
        self.refresh_visuals()

        names = [s.name for s in assigned]
        self._status = (
            f"Random seats for {len(self._agents)} agents "
            f"({names[0]}…{names[-1]}). Still at Spawn — press Run."
        )
        self._log(self._status)

    def _sample_seats(self, seats: list, count: int) -> list:
        """Pick ``count`` unique seats at random (or all seats if fewer available)."""
        n = max(0, min(count, len(seats)))
        if n == 0:
            return []
        return self._rng.sample(list(seats), n)

    @property
    def is_placing_fire(self) -> bool:
        return self._fire.is_placing

    def begin_fire_placement(self) -> None:
        """UI: Start Fire → Cancel. Click viewport to place the cuboid."""
        if self._fire.is_placing:
            return
        ok = self._fire.begin_placement(self._on_fire_placed)
        if ok:
            self._status = "Fire placement — click the viewport to spawn (Cancel to abort)."
        else:
            self._status = "Fire placement failed — no active Viewport."
        self._log(self._status)

    def cancel_fire_placement(self) -> None:
        if not self._fire.is_placing:
            return
        self._fire.cancel_placement()
        self._status = "Fire placement cancelled."
        self._log(self._status)

    def toggle_fire_placement(self) -> None:
        if self._fire.is_placing:
            self.cancel_fire_placement()
        else:
            self.begin_fire_placement()

    def _on_fire_placed(self, position: Vec3) -> None:
        self.trigger_fire_at(position)

    def trigger_fire_at(self, position: Vec3) -> None:
        """Spawn red fire cuboid. Evacuation is perception-driven (FOV + range)."""
        self._fire.trigger_at(position)
        self._exits = scene_util.find_exits()
        exits = [e.name for e in self._exits] or ["(none found)"]
        self._note_state_change(
            "sim",
            "fire placed",
            f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})",
        )
        self._status = (
            f"Fire at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}) — "
            f"agents evacuate when they see it "
            f"(FOV {FIRE_FOV_DEG:.0f}°, {FIRE_VIEW_RANGE_M:.0f}m, wall LOS). "
            f"Exits: {', '.join(exits)}."
        )
        self._log(self._status)

    def trigger_fire(self) -> None:
        """Back-compat: place fire at origin if somehow called without a pick."""
        if self._fire.position is not None:
            self.trigger_fire_at(self._fire.position)
            return
        self.trigger_fire_at((0.0, 0.0, 0.0))

    def _ensure_exit_destination(self, agent: Agent) -> None:
        if not self._exits:
            self._exits = scene_util.find_exits()
        exit_info = scene_util.nearest_exit(agent.position, self._exits)
        if exit_info is None:
            return
        exit_pos = self._navigation.snap_to_navmesh(exit_info.position) or exit_info.position
        if agent.goal != exit_info.name or agent.destination is None or not agent.waypoints:
            agent.goal = exit_info.name
            agent.destination = exit_pos
            agent.waypoints = []
            agent.waypoint_index = 0

    def _separate_walking_agents(self, up: str) -> None:
        """Simple pairwise push. Seated / arrived agents are ignored entirely."""
        movers = [a for a in self._agents if not a.perception.get("at_destination")]
        n = len(movers)
        if n < 2:
            return
        for i in range(n):
            for j in range(i + 1, n):
                a = movers[i]
                b = movers[j]
                pa, pb = a.position, b.position
                dx = pb[0] - pa[0]
                dy = pb[1] - pa[1]
                dz = pb[2] - pa[2]
                if up == "Y":
                    dist_sq = dx * dx + dz * dz
                    dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                else:
                    dist_sq = dx * dx + dy * dy
                    dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0

                min_dist = a.radius + b.radius
                if dist >= min_dist:
                    continue

                if dist < 1e-6:
                    push = min_dist * 0.5
                    a.position = (pa[0] - push, pa[1], pa[2])
                    b.position = (pb[0] + push, pb[1], pb[2])
                    continue

                overlap = (min_dist - dist) * 0.5
                ux, uy, uz = dx / dist, dy / dist, dz / dist
                if up == "Y":
                    a.position = (pa[0] - ux * overlap, pa[1], pa[2] - uz * overlap)
                    b.position = (pb[0] + ux * overlap, pb[1], pb[2] + uz * overlap)
                else:
                    a.position = (pa[0] - ux * overlap, pa[1] - uy * overlap, pa[2])
                    b.position = (pb[0] + ux * overlap, pb[1] + uy * overlap, pb[2])

    def _create_capsule(self, stage, prim_path: str, up: str) -> None:
        if stage.GetPrimAtPath(prim_path):
            stage.RemovePrim(prim_path)
        capsule = UsdGeom.Capsule.Define(stage, prim_path)
        capsule.CreateHeightAttr(max(AGENT_HEIGHT - 2.0 * AGENT_RADIUS, 0.1))
        capsule.CreateRadiusAttr(AGENT_RADIUS)
        capsule.CreateAxisAttr(UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
        # Default debug color: heading to seat (blue)
        UsdGeom.Gprim(capsule.GetPrim()).CreateDisplayColorAttr(
            [Gf.Vec3f(0.15, 0.45, 0.95)]
        )
        xformable = UsdGeom.Xformable(capsule.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    def _clear_agent_prims(self, stage) -> None:
        root = stage.GetPrimAtPath(AGENTS_ROOT)
        if root and root.IsValid():
            to_remove = []
            for child in root.GetChildren():
                name = child.GetName()
                if name.startswith("Agent_") or name in ("TestCube", "AgentFovViz"):
                    to_remove.append(child.GetPath())
            for path in to_remove:
                stage.RemovePrim(path)
        self._agents.clear()
        self._selected_agent_id = None
        self._hovered_agent_id = None
        self._agent_watch.clear()
        self._state_events.clear()
        self._sim_time = 0.0

    def _lift_point(self, floor_point: Vec3, amount: float) -> Vec3:
        up = scene_util.up_axis()
        x, y, z = floor_point
        if up == "Z":
            return (x, y, z + amount)
        return (x, y + amount, z)

    def _ensure_xform(self, stage, path: str) -> None:
        if not stage.GetPrimAtPath(path):
            UsdGeom.Xform.Define(stage, path)

    def _log(self, message: str) -> None:
        carb.log_info(f"[crowd.agents] {message}")
        self._log_lines.append(message)
        if len(self._log_lines) > 50:
            self._log_lines = self._log_lines[-50:]

    def _note_state_change(
        self, agent_id: str, kind: str, detail: str = ""
    ) -> None:
        """Record a discrete state change (not continuous pose updates)."""
        detail = (detail or "").strip()
        if detail:
            line = f"t={self._sim_time:.1f}s  {agent_id}: {kind} — {detail}"
        else:
            line = f"t={self._sim_time:.1f}s  {agent_id}: {kind}"
        self._state_events.append(
            {"agent_id": agent_id, "kind": kind, "detail": detail, "line": line}
        )
        if len(self._state_events) > MAX_STATE_EVENTS:
            self._state_events = self._state_events[-MAX_STATE_EVENTS:]

    def _snapshot_agent_watch(self, agent: Agent, brain_intent: str) -> Dict[str, object]:
        return {
            "brain": brain_intent,
            "goal": agent.goal or "",
            "seat": agent.assigned_seat or "",
            "fire": bool(
                agent.memory.get("fire_seen") or agent.perception.get("fire_visible")
            ),
            "arrived": bool(agent.perception.get("at_destination")),
        }

    def _seed_agent_watch(self, agent: Agent, brain_intent: str = "IDLE") -> None:
        self._agent_watch[agent.agent_id] = self._snapshot_agent_watch(
            agent, brain_intent
        )

    def _track_agent_state_changes(self, agent: Agent, brain_intent: str) -> None:
        cur = self._snapshot_agent_watch(agent, brain_intent)
        prev = self._agent_watch.get(agent.agent_id)
        if prev is None:
            self._agent_watch[agent.agent_id] = cur
            return

        aid = agent.agent_id
        if cur["fire"] and not prev["fire"]:
            self._note_state_change(aid, "saw fire")
        if cur["goal"] != prev["goal"]:
            old_g = prev["goal"] or "—"
            new_g = cur["goal"] or "—"
            self._note_state_change(aid, "goal", f"{old_g} → {new_g}")
        if cur["seat"] != prev["seat"]:
            old_s = prev["seat"] or "—"
            new_s = cur["seat"] or "—"
            self._note_state_change(aid, "seat", f"{old_s} → {new_s}")
        if cur["brain"] != prev["brain"]:
            self._note_state_change(
                aid, "intent", f"{prev['brain']} → {cur['brain']}"
            )
        if cur["arrived"] and not prev["arrived"]:
            where = cur["goal"] or "destination"
            self._note_state_change(aid, "arrived", str(where))
        elif prev["arrived"] and not cur["arrived"]:
            self._note_state_change(aid, "left destination")

        self._agent_watch[agent.agent_id] = cur
