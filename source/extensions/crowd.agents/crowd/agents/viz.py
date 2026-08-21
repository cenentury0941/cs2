# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Debug visuals: agent highlight colors and selected-agent FOV wireframe."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, UsdGeom

from .fire import EYE_HEIGHT_FRAC, FIRE_FOV_DEG, FIRE_VIEW_RANGE_M

Vec3 = Tuple[float, float, float]

FOV_VIZ_PATH = "/World/CrowdAgents/AgentFovViz"
FOV_ARC_SEGMENTS = 48
FOV_LINE_WIDTH = 0.04

# Capsule display colors
COLOR_SELECTED = Gf.Vec3f(0.10, 0.82, 0.28)  # green
COLOR_HOVERED = Gf.Vec3f(0.55, 0.95, 0.60)  # light green
COLOR_TO_SEAT = Gf.Vec3f(0.15, 0.45, 0.95)
COLOR_TO_EXIT = Gf.Vec3f(0.92, 0.12, 0.08)
COLOR_FOV_WIRE = Gf.Vec3f(0.20, 0.95, 0.35)


def agent_highlight_color(
    *,
    selected: bool,
    hovered: bool,
    evacuating: bool,
) -> Gf.Vec3f:
    if selected:
        return COLOR_SELECTED
    if hovered:
        return COLOR_HOVERED
    if evacuating:
        return COLOR_TO_EXIT
    return COLOR_TO_SEAT


def fov_sector_polyline(
    feet: Vec3,
    facing: Vec3,
    *,
    up: str = "Y",
    height: float = 1.7,
    fov_deg: float = FIRE_FOV_DEG,
    range_m: float = FIRE_VIEW_RANGE_M,
    segments: int = FOV_ARC_SEGMENTS,
) -> List[Gf.Vec3f]:
    """Closed wire outline of the horizontal FOV sector at eye height.

    Path: eye → left rim → arc to right rim → eye.
    """
    eye = _lift(feet, height * EYE_HEIGHT_FRAC, up)
    face = _horizontal_unit(facing, up)
    if face is None:
        # Default look along +forward in the horizontal plane
        face = (0.0, 1.0)

    half = math.radians(max(1.0, fov_deg) * 0.5)
    # Yaw from +forward axis in the horizontal plane
    base = math.atan2(face[0], face[1])

    left = _point_at_yaw(eye, base - half, range_m, up)
    right = _point_at_yaw(eye, base + half, range_m, up)

    points: List[Gf.Vec3f] = [Gf.Vec3f(*eye), Gf.Vec3f(*left)]
    for i in range(1, segments):
        t = -half + (2.0 * half) * (i / float(segments))
        points.append(Gf.Vec3f(*_point_at_yaw(eye, base + t, range_m, up)))
    points.append(Gf.Vec3f(*right))
    points.append(Gf.Vec3f(*eye))
    return points


def sync_fov_wireframe(
    agent_feet: Optional[Vec3],
    facing: Optional[Vec3],
    *,
    up: str = "Y",
    height: float = 1.7,
) -> None:
    """Show or hide the selected-agent FOV BasisCurves wireframe."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    if agent_feet is None or facing is None:
        _hide_fov(stage)
        return

    points = fov_sector_polyline(agent_feet, facing, up=up, height=height)
    _ensure_fov_curves(stage, points)
    try:
        omni.usd.get_context().set_pickable(FOV_VIZ_PATH, False)
    except Exception:  # noqa: BLE001
        pass


def clear_fov_wireframe() -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    _hide_fov(stage)


def _ensure_fov_curves(stage, points: List[Gf.Vec3f]) -> None:
    agents_root = "/World/CrowdAgents"
    if not stage.GetPrimAtPath("/World"):
        UsdGeom.Xform.Define(stage, "/World")
    if not stage.GetPrimAtPath(agents_root):
        UsdGeom.Xform.Define(stage, agents_root)

    prim = stage.GetPrimAtPath(FOV_VIZ_PATH)
    if prim and prim.IsValid() and prim.GetTypeName() == "BasisCurves":
        curves = UsdGeom.BasisCurves(prim)
    else:
        if prim and prim.IsValid():
            stage.RemovePrim(FOV_VIZ_PATH)
        curves = UsdGeom.BasisCurves.Define(stage, FOV_VIZ_PATH)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        try:
            curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        except Exception:  # noqa: BLE001
            pass

    n = len(points)
    curves.CreatePointsAttr().Set(points)
    curves.CreateCurveVertexCountsAttr().Set([n])
    try:
        curves.CreateWidthsAttr().Set([FOV_LINE_WIDTH] * n)
    except Exception:  # noqa: BLE001
        pass

    gprim = UsdGeom.Gprim(curves.GetPrim())
    color_attr = gprim.GetDisplayColorAttr()
    if not color_attr:
        color_attr = gprim.CreateDisplayColorAttr()
    color_attr.Set([COLOR_FOV_WIRE])
    try:
        UsdGeom.Imageable(curves.GetPrim()).MakeVisible()
    except Exception:  # noqa: BLE001
        pass


def _hide_fov(stage) -> None:
    prim = stage.GetPrimAtPath(FOV_VIZ_PATH)
    if not prim or not prim.IsValid():
        return
    try:
        UsdGeom.Imageable(prim).MakeInvisible()
    except Exception:  # noqa: BLE001
        try:
            stage.RemovePrim(FOV_VIZ_PATH)
        except Exception:  # noqa: BLE001
            carb.log_warn("[crowd.agents] Could not hide FOV wireframe")


def _lift(floor_point: Vec3, amount: float, up: str) -> Vec3:
    x, y, z = floor_point
    if up == "Z":
        return (x, y, z + amount)
    return (x, y + amount, z)


def _horizontal_unit(facing: Vec3, up: str) -> Optional[Tuple[float, float]]:
    if up == "Z":
        fx, fy = facing[0], facing[1]
    else:
        fx, fy = facing[0], facing[2]
    length = math.sqrt(fx * fx + fy * fy)
    if length < 1e-8:
        return None
    return (fx / length, fy / length)


def _point_at_yaw(origin: Vec3, yaw: float, range_m: float, up: str) -> Vec3:
    """Yaw: atan2(lateral, forward) with forward = +Z (Y-up) or +Y (Z-up)."""
    lateral = math.sin(yaw) * range_m
    forward = math.cos(yaw) * range_m
    if up == "Z":
        return (origin[0] + lateral, origin[1] + forward, origin[2])
    return (origin[0] + lateral, origin[1], origin[2] + forward)
