# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Stage discovery for Spawn zone, Seat.XXX planes, and Exit empties."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import carb
import omni.usd
from pxr import Usd, UsdGeom

Vec3 = Tuple[float, float, float]

SPAWN_NAME = "Spawn"
SEAT_PATTERN = re.compile(r"^Seat\.(\d{3})$")
EXIT_NAMES = ("Exit_A", "Exit_B")


@dataclass
class SeatInfo:
    name: str
    prim_path: str
    position: Vec3
    index: int


@dataclass
class ExitInfo:
    name: str
    prim_path: str
    position: Vec3


@dataclass
class SpawnZone:
    prim_path: str
    min_corner: Vec3
    max_corner: Vec3


def get_stage():
    return omni.usd.get_context().get_stage()


def up_axis(stage=None) -> str:
    stage = stage or get_stage()
    if stage is None:
        return "Y"
    meta = stage.GetMetadata("upAxis")
    return str(meta) if meta is not None else "Y"


def find_prims_named(stage, name: str) -> List[Usd.Prim]:
    matches = []
    for prim in stage.Traverse():
        if prim.GetName() == name:
            matches.append(prim)
    return matches


def world_translation(prim: Usd.Prim) -> Optional[Vec3]:
    if not prim or not prim.IsValid():
        return None
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None
    mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


def world_bbox(prim: Usd.Prim) -> Optional[Tuple[Vec3, Vec3]]:
    """Return (min, max) world AABB, or None."""
    if not prim or not prim.IsValid():
        return None
    boundable = UsdGeom.Boundable(prim)
    if boundable:
        try:
            bbox = boundable.ComputeWorldBound(Usd.TimeCode.Default())
            if hasattr(bbox, "ComputeAlignedRange"):
                rng = bbox.ComputeAlignedRange()
            elif hasattr(bbox, "GetRange"):
                rng = bbox.GetRange()
            else:
                rng = bbox.GetBox()
            mn = rng.GetMin()
            mx = rng.GetMax()
            return (
                (float(mn[0]), float(mn[1]), float(mn[2])),
                (float(mx[0]), float(mx[1]), float(mx[2])),
            )
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] world_bbox failed for {prim.GetPath()}: {exc}")

    t = world_translation(prim)
    if t is None:
        return None
    eps = 0.05
    return (
        (t[0] - eps, t[1] - eps, t[2] - eps),
        (t[0] + eps, t[1] + eps, t[2] + eps),
    )


def find_spawn_zone(stage=None) -> Optional[SpawnZone]:
    stage = stage or get_stage()
    if stage is None:
        return None
    prims = find_prims_named(stage, SPAWN_NAME)
    if not prims:
        return None
    prim = prims[0]
    bbox = world_bbox(prim)
    if bbox is None:
        return None
    return SpawnZone(prim_path=str(prim.GetPath()), min_corner=bbox[0], max_corner=bbox[1])


def find_seats(stage=None) -> List[SeatInfo]:
    stage = stage or get_stage()
    if stage is None:
        return []
    seats: List[SeatInfo] = []
    for prim in stage.Traverse():
        match = SEAT_PATTERN.match(prim.GetName())
        if not match:
            continue
        pos = world_translation(prim)
        if pos is None:
            bbox = world_bbox(prim)
            if bbox:
                mn, mx = bbox
                pos = (
                    0.5 * (mn[0] + mx[0]),
                    0.5 * (mn[1] + mx[1]),
                    0.5 * (mn[2] + mx[2]),
                )
        if pos is None:
            continue
        seats.append(
            SeatInfo(
                name=prim.GetName(),
                prim_path=str(prim.GetPath()),
                position=pos,
                index=int(match.group(1)),
            )
        )
    seats.sort(key=lambda s: s.index)
    return seats


def find_exits(stage=None) -> List[ExitInfo]:
    stage = stage or get_stage()
    if stage is None:
        return []
    exits: List[ExitInfo] = []
    for name in EXIT_NAMES:
        for prim in find_prims_named(stage, name):
            pos = world_translation(prim)
            if pos is None:
                continue
            exits.append(ExitInfo(name=name, prim_path=str(prim.GetPath()), position=pos))
            break
    return exits


def sample_in_spawn(zone: SpawnZone, stage=None, rng: Optional[random.Random] = None) -> Vec3:
    """Sample a point inside the Spawn plane's horizontal (x,y or x,z) bounds."""
    rng = rng or random
    stage = stage or get_stage()
    up = up_axis(stage)
    mn, mx = zone.min_corner, zone.max_corner

    def _u(a: float, b: float) -> float:
        lo, hi = (a, b) if a <= b else (b, a)
        if abs(hi - lo) < 1e-6:
            return lo
        return rng.uniform(lo, hi)

    if up == "Z":
        # Blender-style Z-up: horizontal plane is X/Y (matches "x,y bounds")
        return (_u(mn[0], mx[0]), _u(mn[1], mx[1]), 0.5 * (mn[2] + mx[2]))
    # Y-up: horizontal is X/Z
    return (_u(mn[0], mx[0]), 0.5 * (mn[1] + mx[1]), _u(mn[2], mx[2]))


def point_in_spawn_horizontal(point: Vec3, zone: SpawnZone, stage=None, margin: float = 0.0) -> bool:
    stage = stage or get_stage()
    up = up_axis(stage)
    mn, mx = zone.min_corner, zone.max_corner

    def _inside(v: float, a: float, b: float) -> bool:
        lo, hi = (a, b) if a <= b else (b, a)
        return (lo - margin) <= v <= (hi + margin)

    if up == "Z":
        return _inside(point[0], mn[0], mx[0]) and _inside(point[1], mn[1], mx[1])
    return _inside(point[0], mn[0], mx[0]) and _inside(point[2], mn[2], mx[2])


def nearest_exit(position: Vec3, exits: List[ExitInfo]) -> Optional[ExitInfo]:
    if not exits:
        return None
    best = None
    best_d = float("inf")
    for exit_info in exits:
        p = exit_info.position
        d = math.sqrt(
            (p[0] - position[0]) ** 2
            + (p[1] - position[1]) ** 2
            + (p[2] - position[2]) ** 2
        )
        if d < best_d:
            best_d = d
            best = exit_info
    return best


def describe_scene(stage=None) -> Dict[str, object]:
    stage = stage or get_stage()
    spawn = find_spawn_zone(stage)
    seats = find_seats(stage)
    exits = find_exits(stage)
    return {
        "spawn": spawn.prim_path if spawn else None,
        "seat_count": len(seats),
        "seat_names": [s.name for s in seats],
        "exits": [e.name for e in exits],
        "up_axis": up_axis(stage),
    }
