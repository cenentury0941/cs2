# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Agent state model and USD transform sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import omni.usd
from pxr import Gf, UsdGeom


@dataclass
class Agent:
    """One simulated person: state in Python, pose mirrored to a USD prim."""

    agent_id: str
    prim_path: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: Optional[str] = None
    intent: str = "IDLE"
    perception: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    speed_mps: float = 1.4

    def __post_init__(self) -> None:
        self.spawn_position = self.position
        if not self.perception:
            self.perception = {
                "fire_visible": False,
                "seat_known": bool(self.goal),
                "at_destination": False,
            }

    def reset(self) -> None:
        self.position = self.spawn_position
        self.intent = "IDLE"
        self.perception = {
            "fire_visible": False,
            "seat_known": bool(self.goal),
            "at_destination": False,
        }

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

        translate_op.Set(Gf.Vec3d(*self.position))
