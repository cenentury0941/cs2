# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Simulation orchestrator: agents, navigation, fire, and decision engine."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, UsdGeom

from .agent import Agent
from .ai import DecisionEngine, RuleBasedDecisionEngine
from .fire import FireSystem
from .navigation import NavigationSystem
from . import scene as scene_util

TEST_CUBE_PATH = "/World/CrowdAgents/TestCube"
AGENTS_ROOT = "/World/CrowdAgents"
DEFAULT_AGENT_COUNT = 8
MAX_AGENTS = 10
AGENT_HEIGHT = 1.7
AGENT_RADIUS = 0.3
Vec3 = Tuple[float, float, float]


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

    @property
    def status_text(self) -> str:
        return self._status

    def shutdown(self) -> None:
        self._agents.clear()
        self._running = False

    def update(self, dt: float) -> None:
        if not self._running or dt <= 0.0:
            return

        fire_active = self._fire.is_active
        seated = 0
        evacuating = 0

        for agent in self._agents:
            agent.perception["fire_visible"] = self._fire.is_visible_to(agent)
            if fire_active:
                agent.perception["fire_visible"] = True

            intent = self._brain.decide(agent)

            if intent == "EXIT":
                self._ensure_exit_destination(agent)
                if agent.destination is not None and not agent.perception.get("at_destination"):
                    intent = "WALK"
            elif (
                intent == "GO_TO_SEAT"
                and agent.destination is not None
                and not agent.perception.get("at_destination")
            ):
                intent = "WALK"

            agent.intent = intent
            self._navigation.apply_intent(agent, intent, dt)
            agent.sync_to_usd()

            if agent.perception.get("at_destination"):
                if fire_active or (agent.goal or "").startswith("Exit"):
                    evacuating += 1
                else:
                    seated += 1

        total = len(self._agents)
        if fire_active:
            self._status = f"Running (FIRE) — {evacuating}/{total} at exits"
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
                if self._fire.is_active:
                    self._ensure_exit_destination(agent)
                    if agent.destination is not None:
                        agent.perception["at_destination"] = False
                        self._navigation.set_destination(agent, agent.destination)
                elif agent.destination is not None:
                    # Prompted walk: Spawn → assigned Seat_XXX
                    agent.perception["at_destination"] = False
                    self._navigation.set_destination(agent, agent.destination)
                agent.sync_to_usd()
            self._running = True
            self._status = f"Running — {len(self._agents)} agents walking Spawn → seats"
            self._log(self._status)
            return

        self._running = False
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
            agent.sync_to_usd()
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
                "No Seat_XXX prims found (expected Seat_000 … Seat_024)."
            )
            self._log(self._status)
            return

        self._exits = scene_util.find_exits(stage)
        wanted = count if count is not None else DEFAULT_AGENT_COUNT
        wanted = max(1, min(wanted, MAX_AGENTS, len(seats)))

        self._clear_agent_prims(stage)
        self._ensure_xform(stage, "/World")
        self._ensure_xform(stage, AGENTS_ROOT)

        agents: List[Agent] = []
        assigned_seats = seats[:wanted]
        up = scene_util.up_axis(stage)

        for i, seat in enumerate(assigned_seats):
            feet = scene_util.sample_spawn_feet_on_navmesh(
                spawn_zone,
                snap_fn=self._navigation.snap_to_navmesh,
                rng=self._rng,
            )
            if feet is None:
                self._status = (
                    f"Could only place {len(agents)} agents inside Spawn X/Y on NavMesh. "
                    "Ensure the Spawn plane overlaps the baked NavMesh."
                )
                self._log(self._status)
                break

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
            )
            agent.spawn_position = feet
            agent.memory["seat_position"] = list(seat_pos)
            agent.perception["seat_known"] = True
            agent.perception["at_destination"] = False
            # Idle at Spawn until Run — do not plan seat paths yet
            agent.waypoints = []
            agent.waypoint_index = 0
            agent.intent = "IDLE"
            agent.sync_to_usd()
            agents.append(agent)

        self._agents = agents
        if not agents:
            self._status = (
                "Failed to spawn inside Spawn X/Y on NavMesh — "
                "check Spawn overlaps the cyan walkable area."
            )
            self._log(self._status)
            return

        exit_names = [e.name for e in self._exits] or ["(none)"]
        x0, x1 = spawn_zone.x_range
        y0, y1 = spawn_zone.y_range
        self._status = (
            f"Spawned {len(agents)} idle agents in Spawn "
            f"X[{x0:.1f},{x1:.1f}] Y[{y0:.1f},{y1:.1f}] → "
            f"{assigned_seats[0].name}…{assigned_seats[len(agents) - 1].name}. "
            f"Exits: {', '.join(exit_names)}. Press Run to walk to seats."
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
                agent.sync_to_usd()

        self._status = f"Random destinations set for {planned}/{len(self._agents)} agents."
        self._log(self._status)

    def assign_seats(self) -> None:
        """Re-bind current agents to Seat_XXX (paths planned on Run)."""
        if not self._agents:
            self._status = "Spawn Agents first."
            return
        seats = scene_util.find_seats()
        if not seats:
            self._status = "No Seat_XXX prims found."
            return

        for i, agent in enumerate(self._agents):
            seat = seats[i % len(seats)]
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
            agent.sync_to_usd()

        self._status = (
            f"Assigned seats for {len(self._agents)} agents (still at Spawn). Press Run."
        )
        self._log(self._status)

    def trigger_fire(self) -> None:
        self._fire.trigger()
        self._exits = scene_util.find_exits()
        for agent in self._agents:
            agent.perception["fire_visible"] = True
            agent.perception["at_destination"] = False
            self._ensure_exit_destination(agent)
            if agent.destination is not None:
                self._navigation.set_destination(agent, agent.destination)
            agent.sync_to_usd()
        exits = [e.name for e in self._exits] or ["(none found)"]
        self._status = (
            f"Fire triggered — agents evacuating toward {', '.join(exits)} "
            "(Phase 4 VFX still stub)."
        )
        self._log(self._status)

    def _ensure_exit_destination(self, agent: Agent) -> None:
        if not self._exits:
            self._exits = scene_util.find_exits()
        exit_info = scene_util.nearest_exit(agent.position, self._exits)
        if exit_info is None:
            return
        exit_pos = self._navigation.snap_to_navmesh(exit_info.position) or exit_info.position
        # Replan if new exit or no path yet
        if agent.goal != exit_info.name or agent.destination is None or not agent.waypoints:
            agent.goal = exit_info.name
            agent.destination = exit_pos
            agent.waypoints = []
            agent.waypoint_index = 0

    def _create_capsule(self, stage, prim_path: str, up: str) -> None:
        if stage.GetPrimAtPath(prim_path):
            stage.RemovePrim(prim_path)
        capsule = UsdGeom.Capsule.Define(stage, prim_path)
        capsule.CreateHeightAttr(max(AGENT_HEIGHT - 2.0 * AGENT_RADIUS, 0.1))
        capsule.CreateRadiusAttr(AGENT_RADIUS)
        capsule.CreateAxisAttr(UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
        xformable = UsdGeom.Xformable(capsule.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    def _clear_agent_prims(self, stage) -> None:
        root = stage.GetPrimAtPath(AGENTS_ROOT)
        if root and root.IsValid():
            to_remove = []
            for child in root.GetChildren():
                name = child.GetName()
                if name.startswith("Agent_") or name == "TestCube":
                    to_remove.append(child.GetPath())
            for path in to_remove:
                stage.RemovePrim(path)
        self._agents.clear()

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
