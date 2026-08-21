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
from pxr import Gf, Usd, UsdGeom

Vec3 = Tuple[float, float, float]

SPAWN_NAME = "Spawn"
SEAT_PATTERN = re.compile(r"^Seat_(\d{3})$")
EXIT_NAMES = ("Exit_A", "Exit_B")
_MIN_PLANE_EXTENT = 0.25  # meters; below this AABB is treated as degenerate


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
    """Spawn area derived from the Spawn mesh local plane, in world space.

    Samples randomly in the object's local X/Y plane extents (the two largest
    local axes if needed), then converts to world. NavMesh only supplies height.
    """

    prim_path: str
    geom_path: str
    # Local-space AABB of the geometry (object space)
    local_min: Vec3
    local_max: Vec3
    # Indices of the two local axes that span the plane (usually 0,1 = X,Y)
    axis_u: int
    axis_v: int
    axis_n: int
    # World matrix of the geometry prim
    local_to_world: object  # Gf.Matrix4d
    # Cached world AABB (for logging / rough checks)
    world_min: Vec3
    world_max: Vec3

    @property
    def u_range(self) -> Tuple[float, float]:
        a, b = self.local_min[self.axis_u], self.local_max[self.axis_u]
        return (a, b) if a <= b else (b, a)

    @property
    def v_range(self) -> Tuple[float, float]:
        a, b = self.local_min[self.axis_v], self.local_max[self.axis_v]
        return (a, b) if a <= b else (b, a)

    @property
    def area(self) -> float:
        u0, u1 = self.u_range
        v0, v1 = self.v_range
        return max(0.0, u1 - u0) * max(0.0, v1 - v0)


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


def _range_from_bbox3d(bbox) -> Optional[Tuple[Gf.Vec3d, Gf.Vec3d]]:
    try:
        if hasattr(bbox, "ComputeAlignedRange"):
            rng = bbox.ComputeAlignedRange()
        elif hasattr(bbox, "GetRange"):
            rng = bbox.GetRange()
        else:
            rng = bbox.GetBox()
        return rng.GetMin(), rng.GetMax()
    except Exception:  # noqa: BLE001
        return None


def world_bbox(prim: Usd.Prim) -> Optional[Tuple[Vec3, Vec3]]:
    if not prim or not prim.IsValid():
        return None
    boundable = UsdGeom.Boundable(prim)
    if boundable:
        try:
            bbox = boundable.ComputeWorldBound(Usd.TimeCode.Default())
            pair = _range_from_bbox3d(bbox)
            if pair:
                mn, mx = pair
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


def local_bbox(prim: Usd.Prim) -> Optional[Tuple[Vec3, Vec3]]:
    """Local-space AABB of a boundable prim."""
    if not prim or not prim.IsValid():
        return None

    # Prefer authored extent (mesh local)
    for attr_name in ("extent", "primvars:extent"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.HasAuthoredValueOpinion():
            try:
                ext = attr.Get()
                if ext is not None and len(ext) >= 2:
                    mn, mx = ext[0], ext[1]
                    return (
                        (float(mn[0]), float(mn[1]), float(mn[2])),
                        (float(mx[0]), float(mx[1]), float(mx[2])),
                    )
            except Exception:  # noqa: BLE001
                pass

    boundable = UsdGeom.Boundable(prim)
    if not boundable:
        return None
    try:
        bbox = boundable.ComputeLocalBound(Usd.TimeCode.Default())
        pair = _range_from_bbox3d(bbox)
        if pair:
            mn, mx = pair
            return (
                (float(mn[0]), float(mn[1]), float(mn[2])),
                (float(mx[0]), float(mx[1]), float(mx[2])),
            )
    except Exception as exc:  # noqa: BLE001
        carb.log_warn(f"[crowd.agents] local_bbox failed for {prim.GetPath()}: {exc}")
    return None


def _plane_axes(local_min: Vec3, local_max: Vec3) -> Tuple[int, int, int]:
    """Return (axis_u, axis_v, axis_n) — two largest extents + thinnest (normal)."""
    extents = [abs(local_max[i] - local_min[i]) for i in range(3)]
    order = sorted(range(3), key=lambda i: extents[i])
    axis_n = order[0]
    axis_u, axis_v = order[1], order[2]
    # Prefer X,Y as (u,v) when they are the large pair (Blender plane)
    if {axis_u, axis_v} == {0, 1}:
        axis_u, axis_v = 0, 1
    return axis_u, axis_v, axis_n


def _bbox_area(mn: Vec3, mx: Vec3) -> float:
    extents = sorted([abs(mx[i] - mn[i]) for i in range(3)])
    return extents[1] * extents[2]


def _iter_boundable_prims(root: Usd.Prim) -> List[Usd.Prim]:
    out = []
    if not root or not root.IsValid():
        return out
    stack = [root]
    while stack:
        prim = stack.pop()
        if UsdGeom.Boundable(prim):
            out.append(prim)
        stack.extend(list(prim.GetChildren()))
    return out


def _pick_spawn_geometry(spawn_prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Choose the Spawn prim or a descendant with the largest planar area."""
    best = None
    best_area = -1.0
    for prim in _iter_boundable_prims(spawn_prim):
        lb = local_bbox(prim)
        wb = world_bbox(prim)
        area_l = _bbox_area(lb[0], lb[1]) if lb else 0.0
        area_w = _bbox_area(wb[0], wb[1]) if wb else 0.0
        area = max(area_l, area_w)
        if area > best_area:
            best_area = area
            best = prim
    if best is None or best_area < _MIN_PLANE_EXTENT * _MIN_PLANE_EXTENT:
        carb.log_warn(
            f"[crowd.agents] Spawn geometry under {spawn_prim.GetPath()} "
            f"has tiny area ({best_area:.4f}). Check scale is applied."
        )
    return best


def find_spawn_zone(stage=None) -> Optional[SpawnZone]:
    stage = stage or get_stage()
    if stage is None:
        return None
    prims = find_prims_named(stage, SPAWN_NAME)
    if not prims:
        return None
    spawn_root = prims[0]
    geom = _pick_spawn_geometry(spawn_root)
    if geom is None:
        return None

    lb = local_bbox(geom)
    wb = world_bbox(geom)
    if lb is None and wb is not None:
        # Degenerate local — fall back to treating world AABB as local with identity
        lb = wb
    if lb is None:
        return None
    if wb is None:
        wb = lb

    axis_u, axis_v, axis_n = _plane_axes(lb[0], lb[1])
    # Force X,Y when those extents are usable (user contract: object X/Y bounds)
    ext = [abs(lb[1][i] - lb[0][i]) for i in range(3)]
    if ext[0] >= _MIN_PLANE_EXTENT and ext[1] >= _MIN_PLANE_EXTENT:
        axis_u, axis_v, axis_n = 0, 1, 2

    xformable = UsdGeom.Xformable(geom)
    mat = (
        xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if xformable
        else Gf.Matrix4d(1.0)
    )

    zone = SpawnZone(
        prim_path=str(spawn_root.GetPath()),
        geom_path=str(geom.GetPath()),
        local_min=lb[0],
        local_max=lb[1],
        axis_u=axis_u,
        axis_v=axis_v,
        axis_n=axis_n,
        local_to_world=mat,
        world_min=wb[0],
        world_max=wb[1],
    )
    u0, u1 = zone.u_range
    v0, v1 = zone.v_range
    carb.log_info(
        f"[crowd.agents] Spawn zone root={zone.prim_path} geom={zone.geom_path} "
        f"local axes u={axis_u}[{u0:.2f},{u1:.2f}] v={axis_v}[{v0:.2f},{v1:.2f}] "
        f"area={zone.area:.2f}"
    )
    if zone.area < _MIN_PLANE_EXTENT * _MIN_PLANE_EXTENT:
        carb.log_warn(
            "[crowd.agents] Spawn plane area is near zero — "
            "Apply Scale on the Spawn mesh in Blender/USD."
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


def sample_in_spawn_local(zone: SpawnZone, rng: Optional[random.Random] = None) -> Vec3:
    """Random world-space point on the Spawn plane (local X/Y extents)."""
    rng = rng or random
    u0, u1 = zone.u_range
    v0, v1 = zone.v_range
    n0 = zone.local_min[zone.axis_n]
    n1 = zone.local_max[zone.axis_n]
    n_mid = 0.5 * (n0 + n1)

    def _u(lo: float, hi: float) -> float:
        if abs(hi - lo) < 1e-8:
            return lo
        return rng.uniform(lo, hi)

    local = [0.0, 0.0, 0.0]
    local[zone.axis_u] = _u(u0, u1)
    local[zone.axis_v] = _u(v0, v1)
    local[zone.axis_n] = n_mid

    world = zone.local_to_world.Transform(Gf.Vec3d(local[0], local[1], local[2]))
    return (float(world[0]), float(world[1]), float(world[2]))


def sample_in_spawn_xy(zone: SpawnZone, rng: Optional[random.Random] = None) -> Vec3:
    """Back-compat name — samples the Spawn object's plane bounds in local X/Y."""
    return sample_in_spawn_local(zone, rng=rng)


def point_in_spawn_xy(point: Vec3, zone: SpawnZone, margin: float = 0.0) -> bool:
    """Project world point into Spawn local space; test u/v within plane bounds."""
    try:
        inv = zone.local_to_world.GetInverse()
        local = inv.Transform(Gf.Vec3d(point[0], point[1], point[2]))
    except Exception:  # noqa: BLE001
        # Fallback: world AABB on the two largest world axes
        return (
            zone.world_min[0] - margin <= point[0] <= zone.world_max[0] + margin
            and zone.world_min[1] - margin <= point[1] <= zone.world_max[1] + margin
        )

    u0, u1 = zone.u_range
    v0, v1 = zone.v_range
    u = float(local[zone.axis_u])
    v = float(local[zone.axis_v])
    return (u0 - margin) <= u <= (u1 + margin) and (v0 - margin) <= v <= (v1 + margin)


def sample_spawn_feet_on_navmesh(
    zone: SpawnZone,
    snap_fn: Callable[[Vec3], Optional[Vec3]],
    rng: Optional[random.Random] = None,
    attempts: int = 64,
    existing: Optional[List[Vec3]] = None,
    min_spacing: float = 0.7,
) -> Optional[Vec3]:
    """Random point on Spawn plane (local X/Y), snapped onto NavMesh height."""
    rng = rng or random
    existing = existing or []

    for _ in range(attempts):
        guess = sample_in_spawn_local(zone, rng=rng)
        snapped = snap_fn(guess)
        if snapped is None:
            for delta in (0.0, 0.2, -0.2, 0.5, -0.5, 1.0, -1.0):
                # Nudge along world up guesses
                for axis in (2, 1):
                    probe = [guess[0], guess[1], guess[2]]
                    probe[axis] = guess[axis] + delta
                    snapped = snap_fn((probe[0], probe[1], probe[2]))
                    if snapped is not None:
                        break
                if snapped is not None:
                    break
        if snapped is None:
            continue

        # Keep planar (u/v) position from the Spawn sample; take height from NavMesh.
        # Reconstruct by blending: use snapped height axis from stage up.
        up = up_axis()
        if up == "Y":
            feet = (guess[0], snapped[1], guess[2])
        else:
            feet = (guess[0], guess[1], snapped[2])

        # If locking XZ/XY drifted off the plane, prefer full snap if still on plane
        if not point_in_spawn_xy(feet, zone, margin=0.15):
            if point_in_spawn_xy(snapped, zone, margin=0.35):
                feet = snapped
            else:
                continue
        else:
            verify = snap_fn(feet)
            if verify is not None:
                if up == "Y":
                    feet = (feet[0], verify[1], feet[2])
                else:
                    feet = (feet[0], feet[1], verify[2])

        # Spacing vs already placed agents
        ok = True
        for other in existing:
            dx = feet[0] - other[0]
            dy = feet[1] - other[1]
            dz = feet[2] - other[2]
            if (dx * dx + dy * dy + dz * dz) < (min_spacing * min_spacing):
                ok = False
                break
        if not ok:
            continue
        return feet

    return None


def sample_in_spawn(zone: SpawnZone, stage=None, rng: Optional[random.Random] = None) -> Vec3:
    _ = stage
    return sample_in_spawn_local(zone, rng=rng)


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
        "spawn_geom": spawn.geom_path if spawn else None,
        "spawn_area": spawn.area if spawn else None,
        "spawn_uv": (spawn.u_range, spawn.v_range) if spawn else None,
        "seat_count": len(seats),
        "seat_names": [s.name for s in seats],
        "exits": [e.name for e in exits],
        "up_axis": up_axis(stage),
    }
