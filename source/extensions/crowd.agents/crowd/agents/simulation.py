# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Simulation orchestrator: agents, navigation, fire, and decision engine."""

from __future__ import annotations

from typing import List, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, UsdGeom

from .agent import Agent
from .ai import DecisionEngine, RuleBasedDecisionEngine
from .fire import FireSystem
from .navigation import NavigationSystem

TEST_CUBE_PATH = "/World/CrowdAgents/TestCube"
AGENT_PRIM_PATH = "/World/CrowdAgents/Agent_001"
AGENTS_ROOT = "/World/CrowdAgents"
Vec3 = Tuple[float, float, float]


class Simulation:
    """Owns agents and wires decision → navigation → USD each tick."""

    def __init__(self) -> None:
        self._running = False
        self._agents: List[Agent] = []
        self._navigation = NavigationSystem()
        self._fire = FireSystem()
        self._brain: DecisionEngine = RuleBasedDecisionEngine()
        self._status = "Ready — bake NavMesh, spawn an agent, set destination, Run."
        self._log_lines: List[str] = []

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
        for agent in self._agents:
            agent.perception["fire_visible"] = self._fire.is_visible_to(agent)
            if fire_active and not agent.perception.get("fire_visible"):
                agent.perception["fire_visible"] = True

            intent = self._brain.decide(agent)
            if (
                intent == "GO_TO_SEAT"
                and agent.destination is not None
                and not agent.perception.get("at_destination")
            ):
                intent = "WALK"

            agent.intent = intent
            self._navigation.apply_intent(agent, intent, dt)
            agent.sync_to_usd()

        if self._agents:
            a = self._agents[0]
            if a.perception.get("at_destination"):
                self._status = f"{a.agent_id} arrived"
            else:
                p = a.position
                self._status = (
                    f"Running — {a.agent_id} @ ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) "
                    f"intent={a.intent}"
                )

    def set_running(self, running: bool) -> None:
        if running and not self._agents:
            self._status = "Spawn an agent first."
            self._log(self._status)
            return
        if running:
            for agent in self._agents:
                if agent.destination is not None and not agent.waypoints:
                    self._navigation.set_destination(agent, agent.destination)
                if agent.destination is not None:
                    agent.perception["at_destination"] = False
                    agent.goal = agent.goal or "destination"
        self._running = running
        self._status = "Running" if running else "Paused"
        self._log(self._status)

    def reset(self) -> None:
        self._running = False
        self._fire.reset()
        for agent in self._agents:
            agent.reset()
            agent.sync_to_usd()
        self._status = "Reset"
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
            self._status = "NavMesh bake OK — spawn an agent."
        else:
            self._status = (
                "NavMesh bake failed — create/scale Include Volume over the floor, "
                "then try again (or bake from Window > Navigation > NavMesh)."
            )
        self._log(self._status)

    def spawn_agents(self) -> None:
        """Phase 2: spawn a single capsule agent snapped to the NavMesh."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status = "No stage open."
            return

        if not self._navigation.has_navmesh():
            # Try bake once; user may already have cyan mesh from UI bake
            if not self._navigation.generate_navmesh() and not self._navigation.has_navmesh():
                self._status = "No NavMesh — bake first (Generate NavMesh)."
                self._log(self._status)
                return

        spawn = self._pick_spawn_point()
        if spawn is None:
            self._status = "Could not find a NavMesh spawn point."
            self._log(self._status)
            return

        self._ensure_xform(stage, "/World")
        self._ensure_xform(stage, AGENTS_ROOT)
        if stage.GetPrimAtPath(AGENT_PRIM_PATH):
            stage.RemovePrim(AGENT_PRIM_PATH)

        height = 1.7
        radius = 0.3
        capsule = UsdGeom.Capsule.Define(stage, AGENT_PRIM_PATH)
        capsule.CreateHeightAttr(max(height - 2.0 * radius, 0.1))
        capsule.CreateRadiusAttr(radius)
        up = self._up_axis(stage)
        capsule.CreateAxisAttr(UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)

        # Logical position = feet on NavMesh; sync_to_usd lifts to capsule center.
        agent = Agent(
            agent_id="Agent_001",
            prim_path=AGENT_PRIM_PATH,
            position=spawn,
            goal="destination",
            height=height,
        )
        agent.spawn_position = spawn
        xformable = UsdGeom.Xformable(capsule.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*agent.center_position()))
        agent.sync_to_usd()
        self._agents = [agent]

        self._status = (
            f"Spawned {agent.agent_id} feet=({spawn[0]:.1f}, {spawn[1]:.1f}, {spawn[2]:.1f}) "
            "— set destination, then Run."
        )
        self._log(self._status)

    def set_random_destination(self) -> None:
        if not self._agents:
            self._status = "Spawn an agent first."
            return

        agent = self._agents[0]
        dest = self._navigation.random_point(seed_key="crowd.dest")
        if dest is None:
            dest = self._offset_on_ground(agent.position, 5.0, 0.0)
            snapped = self._navigation.snap_to_navmesh(dest)
            dest = snapped or dest

        # Keep planning in feet space (NavMesh), never write floor into capsule center.
        ok = self._navigation.set_destination(agent, dest)
        agent.goal = "destination"
        agent.sync_to_usd()
        if ok:
            d = agent.destination or dest
            self._status = (
                f"Destination ({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}) — "
                f"{len(agent.waypoints)} waypoints. Press Run."
            )
        else:
            self._status = "Failed to set destination."
        self._log(self._status)

    def trigger_fire(self) -> None:
        self._fire.trigger()
        self._status = "Fire triggered (stub VFX — Phase 4)."
        self._log(self._status)

    def _pick_spawn_point(self) -> Optional[Vec3]:
        point = self._navigation.random_point(seed_key="crowd.spawn")
        if point is not None:
            return point
        # Try origin / common floor guesses
        for guess in ((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.1, 0.0), (1.0, 0.0, 1.0)):
            snapped = self._navigation.snap_to_navmesh(guess)
            if snapped is not None:
                return snapped
        return None

    def _up_axis(self, stage) -> str:
        up = stage.GetMetadata("upAxis")
        if up is None:
            return "Y"
        return str(up)

    def _lift_point(self, floor_point: Vec3, amount: float) -> Vec3:
        stage = omni.usd.get_context().get_stage()
        up = self._up_axis(stage) if stage else "Y"
        x, y, z = floor_point
        if up == "Z":
            return (x, y, z + amount)
        return (x, y + amount, z)

    def _offset_on_ground(self, point: Vec3, dx: float, dz_or_dy: float) -> Vec3:
        stage = omni.usd.get_context().get_stage()
        up = self._up_axis(stage) if stage else "Y"
        x, y, z = point
        if up == "Z":
            return (x + dx, y + dz_or_dy, z)
        return (x + dx, y, z + dz_or_dy)

    def _ensure_xform(self, stage, path: str) -> None:
        if not stage.GetPrimAtPath(path):
            UsdGeom.Xform.Define(stage, path)

    def _log(self, message: str) -> None:
        carb.log_info(f"[crowd.agents] {message}")
        self._log_lines.append(message)
        if len(self._log_lines) > 50:
            self._log_lines = self._log_lines[-50:]
