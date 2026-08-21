# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Agent state model and USD transform sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

import omni.usd
from pxr import Gf, UsdGeom

Vec3 = Tuple[float, float, float]


@dataclass
class Agent:
    """One simulated person.

    ``position`` is the **feet** point on the NavMesh. USD sync lifts by
    ``height / 2`` so the capsule center stands on the floor.
    """

    agent_id: str
    prim_path: str
    position: Vec3 = (0.0, 0.0, 0.0)
    spawn_position: Vec3 = (0.0, 0.0, 0.0)
    goal: Optional[str] = None
    destination: Optional[Vec3] = None
    assigned_seat: Optional[str] = None
    seat_prim_path: Optional[str] = None
    intent: str = "IDLE"
    perception: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    waypoints: List[Vec3] = field(default_factory=list)
    waypoint_index: int = 0
    speed_mps: float = 1.0
    height: float = 1.7
    radius: float = 0.35
    retry_timer: float = 0.0
    last_move_pos: Vec3 = (0.0, 0.0, 0.0)
    last_waypoint_dist: float = 0.0
    last_wp_index: int = -1
    # Stable corridor lane preference in [-1, 1] (0 ≈ center); set at spawn / plan
    lane_bias: float = 0.0
    # Horizontal look direction (world); used for fire FOV. Y-up: (x, 0, z)
    facing: Vec3 = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        self.spawn_position = self.position
        self.last_move_pos = self.position
        if not self.perception:
            self.perception = {
                "fire_visible": False,
                "smoke_visible": False,
                "seat_known": bool(self.goal),
                "at_destination": False,
            }

    @property
    def is_seated(self) -> bool:
        return bool(self.perception.get("at_destination")) and not (
            self.goal or ""
        ).startswith("Exit")

    def face_toward(self, target: Vec3, up: str = "Y") -> None:
        """Point facing along the horizontal vector to ``target`` when possible."""
        if up == "Z":
            dx, dy = target[0] - self.position[0], target[1] - self.position[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-8:
                self.facing = (dx / length, dy / length, 0.0)
        else:
            dx, dz = target[0] - self.position[0], target[2] - self.position[2]
            length = math.sqrt(dx * dx + dz * dz)
            if length > 1e-8:
                self.facing = (dx / length, 0.0, dz / length)

    def reset(self) -> None:
        self.position = self.spawn_position
        self.intent = "IDLE"
        self.waypoints = []
        self.waypoint_index = 0
        self.retry_timer = 0.0
        self.last_move_pos = self.spawn_position
        self.last_waypoint_dist = 0.0
        self.last_wp_index = -1
        self.memory.pop("fire_seen", None)
        if "lane_bias" in self.memory:
            self.lane_bias = float(self.memory["lane_bias"])
        if self.assigned_seat and self.memory.get("seat_position"):
            self.goal = self.assigned_seat
            self.destination = tuple(self.memory["seat_position"])  # type: ignore[assignment]
            seat_pos = self.destination
            if seat_pos is not None:
                self.face_toward(seat_pos)
        else:
            self.destination = None
        self.perception = {
            "fire_visible": False,
            "smoke_visible": False,
            "seat_known": bool(self.assigned_seat or self.goal),
            "at_destination": False,
        }

    def center_position(self) -> Vec3:
        return _lift(self.position, self.height * 0.5)

    def sync_to_usd(self, *, highlight: Optional[str] = None) -> None:
        """Write feet→center transform and debug color.

        ``highlight``: ``\"selected\"`` | ``\"hovered\"`` | ``None``.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()

        translate_op.Set(Gf.Vec3d(*self.center_position()))
        self._sync_debug_color(prim, highlight=highlight)

    def _sync_debug_color(self, prim, *, highlight: Optional[str] = None) -> None:
        from .viz import agent_highlight_color

        evacuating = (self.goal or "").startswith("Exit")
        color = agent_highlight_color(
            selected=(highlight == "selected"),
            hovered=(highlight == "hovered"),
            evacuating=evacuating,
        )
        gprim = UsdGeom.Gprim(prim)
        attr = gprim.GetDisplayColorAttr()
        if not attr:
            attr = gprim.CreateDisplayColorAttr()
        attr.Set([color])

def _lift(floor_point: Vec3, amount: float) -> Vec3:
    stage = omni.usd.get_context().get_stage()
    up = "Y"
    if stage is not None:
        meta = stage.GetMetadata("upAxis")
        if meta is not None:
            up = str(meta)
    x, y, z = floor_point
    if up == "Z":
        return (x, y, z + amount)
    return (x, y + amount, z)
