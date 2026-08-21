# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Stage discovery for Spawn zone, Seat_XXX planes, and Exit empties."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import carb
import omni.usd
from pxr import Usd, UsdGeom

Vec3 = Tuple[float, float, float]

SPAWN_NAME = "Spawn"
SEAT_PATTERN = re.compile(r"^Seat_(\d{3})$")
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
    """World AABB of the Spawn plane.

    Agents spawn at random **world X/Y** inside this rectangle (Blender Z-up
    plane), then feet height is taken from the NavMesh. Seats are separate.
    """

    prim_path: str
    min_corner: Vec3
    max_corner: Vec3

    @property
    def x_range(self) -> Tuple[float, float]:
        a, b = self.min_corner[0], self.max_corner[0]
        return (a, b) if a <= b else (b, a)

    @property
    def y_range(self) -> Tuple[float, float]:
        a, b = self.min_corner[1], self.max_corner[1]
        return (a, b) if a <= b else (b, a)

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.min_corner[2] + self.max_corner[2])


def get_stage():
    return omni.usd.get_context().get_stage()


def up_axis(stage=None) -> str:
    stage = stage or get_stage()
    if stage is None:
        return "Z"
    meta = stage.GetMetadata("upAxis")
    return str(meta) if meta is not None else "Z"


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
    zone = SpawnZone(prim_path=str(prim.GetPath()), min_corner=bbox[0], max_corner=bbox[1])
    carb.log_info(
        f"[crowd.agents] Spawn zone {zone.prim_path} "
        f"X=[{zone.x_range[0]:.2f},{zone.x_range[1]:.2f}] "
        f"Y=[{zone.y_range[0]:.2f},{zone.y_range[1]:.2f}]"
    )
    return zone


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


def sample_in_spawn_xy(zone: SpawnZone, rng: Optional[random.Random] = None) -> Vec3:
    """Random point in Spawn's world X/Y rectangle, Z at plane mid-height."""
    rng = rng or random
    x0, x1 = zone.x_range
    y0, y1 = zone.y_range

    def _u(lo: float, hi: float) -> float:
        if abs(hi - lo) < 1e-6:
            return lo
        return rng.uniform(lo, hi)

    return (_u(x0, x1), _u(y0, y1), zone.z_mid)


def point_in_spawn_xy(point: Vec3, zone: SpawnZone, margin: float = 0.0) -> bool:
    """True if point's world X/Y lie inside the Spawn rectangle."""
    x0, x1 = zone.x_range
    y0, y1 = zone.y_range
    return (x0 - margin) <= point[0] <= (x1 + margin) and (y0 - margin) <= point[1] <= (
        y1 + margin
    )


def sample_spawn_feet_on_navmesh(
    zone: SpawnZone,
    snap_fn: Callable[[Vec3], Optional[Vec3]],
    rng: Optional[random.Random] = None,
    attempts: int = 48,
) -> Optional[Vec3]:
    """Pick random Spawn X/Y, snap to NavMesh, keep X/Y inside Spawn.

    Height (Z) comes from the NavMesh so feet sit on walkable geometry.
    """
    rng = rng or random
    for _ in range(attempts):
        guess = sample_in_spawn_xy(zone, rng=rng)
        snapped = snap_fn(guess)
        if snapped is None:
            # Try a few vertical probes in case the plane sits slightly off the mesh
            for dz in (0.0, 0.25, -0.25, 0.5, -0.5, 1.0):
                snapped = snap_fn((guess[0], guess[1], guess[2] + dz))
                if snapped is not None:
                    break
        if snapped is None:
            continue

        # Stay in Spawn X/Y; only adopt NavMesh height
        feet = (guess[0], guess[1], snapped[2])
        if not point_in_spawn_xy(feet, zone, margin=0.05):
            continue

        # Confirm the locked X/Y still projects onto the mesh
        verify = snap_fn(feet)
        if verify is None:
            continue
        # If snap pulls far in X/Y, reject (Spawn must overlap NavMesh)
        dx = verify[0] - feet[0]
        dy = verify[1] - feet[1]
        if (dx * dx + dy * dy) > (0.75 * 0.75):
            continue
        return (feet[0], feet[1], verify[2])

    return None


# Back-compat aliases used by older tests
def sample_in_spawn(zone: SpawnZone, stage=None, rng: Optional[random.Random] = None) -> Vec3:
    _ = stage
    return sample_in_spawn_xy(zone, rng=rng)


def point_in_spawn_horizontal(
    point: Vec3, zone: SpawnZone, stage=None, margin: float = 0.0
) -> bool:
    _ = stage
    return point_in_spawn_xy(point, zone, margin=margin)


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
        "spawn_xy": (spawn.x_range, spawn.y_range) if spawn else None,
        "seat_count": len(seats),
        "seat_names": [s.name for s in seats],
        "exits": [e.name for e in exits],
        "up_axis": up_axis(stage),
    }
