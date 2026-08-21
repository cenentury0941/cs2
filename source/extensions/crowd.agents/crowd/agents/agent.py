# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Agent state model and USD transform sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    speed_mps: float = 1.4
    height: float = 1.7
    radius: float = 0.35
    retry_timer: float = 0.0
    last_move_pos: Vec3 = (0.0, 0.0, 0.0)
    last_waypoint_dist: float = 0.0
    last_wp_index: int = -1
    # Stable corridor lane preference in [-1, 1] (0 ≈ center); set at spawn / plan
    lane_bias: float = 0.0

    def __post_init__(self) -> None:
        self.spawn_position = self.position
        self.last_move_pos = self.position
        if not self.perception:
            self.perception = {
                "fire_visible": False,
                "seat_known": bool(self.goal),
                "at_destination": False,
            }

    @property
    def is_seated(self) -> bool:
        return bool(self.perception.get("at_destination")) and not (
            self.goal or ""
        ).startswith("Exit")

    def reset(self) -> None:
        self.position = self.spawn_position
        self.intent = "IDLE"
        self.waypoints = []
        self.waypoint_index = 0
        self.retry_timer = 0.0
        self.last_move_pos = self.spawn_position
        self.last_waypoint_dist = 0.0
        self.last_wp_index = -1
        if "lane_bias" in self.memory:
            self.lane_bias = float(self.memory["lane_bias"])
        if self.assigned_seat and self.memory.get("seat_position"):
            self.goal = self.assigned_seat
            self.destination = tuple(self.memory["seat_position"])  # type: ignore[assignment]
        else:
            self.destination = None
        self.perception = {
            "fire_visible": False,
            "seat_known": bool(self.assigned_seat or self.goal),
            "at_destination": False,
        }

    def center_position(self) -> Vec3:
        return _lift(self.position, self.height * 0.5)

    def sync_to_usd(self) -> None:
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
