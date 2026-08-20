# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Simulation orchestrator: agents, navigation, fire, and decision engine."""

from __future__ import annotations

from typing import List, Tuple

import carb
import omni.usd
from pxr import Gf, UsdGeom

from .agent import Agent
from .ai import DecisionEngine, RuleBasedDecisionEngine
from .fire import FireSystem
from .navigation import NavigationSystem

TEST_CUBE_PATH = "/World/CrowdAgents/TestCube"
AGENTS_ROOT = "/World/CrowdAgents"


class Simulation:
    """Owns agents and wires decision → navigation → USD each tick."""

    def __init__(self) -> None:
        self._running = False
        self._agents: List[Agent] = []
        self._navigation = NavigationSystem()
        self._fire = FireSystem()
        self._brain: DecisionEngine = RuleBasedDecisionEngine()
        self._status = "Ready — open a venue or spawn a test cube."
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
                # Global emergency flag for MVP; refine with volumes later.
                agent.perception["fire_visible"] = True

            intent = self._brain.decide(agent)
            agent.intent = intent
            self._navigation.apply_intent(agent, intent, dt)
            agent.sync_to_usd()

    def set_running(self, running: bool) -> None:
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
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.5))

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
        self._status = (
            "NavMesh generate requested (stub — Phase 2)."
            if ok
            else "NavMesh generation not available yet."
        )
        self._log(self._status)

    def spawn_agents(self) -> None:
        self._status = "Spawn Agents stub — Phase 3 (need NavMesh + seats)."
        self._log(self._status)

    def trigger_fire(self) -> None:
        self._fire.trigger()
        self._status = "Fire triggered (stub VFX — Phase 4)."
        self._log(self._status)

    def _ensure_xform(self, stage, path: str) -> None:
        if not stage.GetPrimAtPath(path):
            UsdGeom.Xform.Define(stage, path)

    def _log(self, message: str) -> None:
        carb.log_info(f"[crowd.agents] {message}")
        self._log_lines.append(message)
        if len(self._log_lines) > 50:
            self._log_lines = self._log_lines[-50:]
