# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""NavMesh baking and path following via omni.anim.navigation.core."""

from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

import carb
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

if TYPE_CHECKING:
    from .agent import Agent

Vec3 = Tuple[float, float, float]
WAYPOINT_EPSILON = 0.20
GOAL_ARRIVAL_EPSILON = 0.35
MOVE_EPS = 0.02  # m per tick — below this counts as "not moving"
RETRY_TIMEOUT_S = 1.0
# Replan when closing on the *current waypoint* slower than this (m/s)
MIN_WP_CLOSING_MPS = 0.25
# Cap dt used by replan so a hitch can't force every agent to replan next frame
REPLAN_DT_CAP_S = 0.1
# Blender Nav.001 → USD Nav_001 (also accept Nav.001)
NAV_MARKER_PATTERN = re.compile(r"^Nav[._](\d+)$", re.IGNORECASE)
NAVMESH_VOLUME_INCLUDE = 0

# Light lane jitter — capped by NavMesh bake agent radius (no side probes)
LANE_RADIUS_SCALE = 0.85  # fraction of bake radius used as max lateral offset
DEFAULT_BAKE_RADIUS = 0.4


class NavigationSystem:
    """Wraps Omniverse NavMesh bake + shortest-path queries."""

    def __init__(self) -> None:
        self._interface = None
        self._random_counter = 0
        self._lane_rng = random.Random(41)

    def _acquire(self):
        if self._interface is not None:
            return self._interface
        try:
            import omni.anim.navigation.core as nav

            self._interface = nav.acquire_interface()
            return self._interface
        except Exception as exc:  # noqa: BLE001
            carb.log_error(f"[crowd.agents] NavMesh interface unavailable: {exc}")
            return None

    def has_navmesh(self) -> bool:
        inav = self._acquire()
        if inav is None:
            return False
        try:
            return inav.get_navmesh() is not None
        except Exception:  # noqa: BLE001
            return False

    def ensure_include_volume(self) -> bool:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if type_name and "NavMeshVolume" in type_name:
                return True
            if "navmesh" in prim.GetName().lower() and "volume" in prim.GetName().lower():
                return True

        try:
            omni.kit.commands.execute(
                "CreateNavMeshVolumeCommand",
                parent_prim_path=Sdf.Path("/World"),
                volume_type=0,
            )
            carb.log_info("[crowd.agents] Created NavMesh Include Volume under /World")
            return True
        except Exception as exc:  # noqa: BLE001
            carb.log_error(f"[crowd.agents] Failed to create NavMesh volume: {exc}")
            return False

    def generate_navmesh(self) -> bool:
        inav = self._acquire()
        if inav is None:
            return False

        self.ensure_include_volume()
        try:
            inav.start_navmesh_baking_and_wait()
            navmesh = inav.get_navmesh()
            if navmesh is None:
                carb.log_warn(
                    "[crowd.agents] Bake finished but get_navmesh() is None — "
                    "scale Include Volume over walkable mesh and check agent size units."
                )
                return False
            carb.log_info("[crowd.agents] NavMesh bake succeeded")
            return True
        except Exception as exc:  # noqa: BLE001
            carb.log_error(f"[crowd.agents] NavMesh bake failed: {exc}")
            return False

    def create_include_volumes_from_nav_markers(self) -> Tuple[int, str]:
        """Replace Nav_XXX / Nav.XXX cubes with NavMesh Include Volumes.

        Copies each marker's world pose and size, deletes the marker, and creates
        an Include volume under ``/World/NavVolumes`` with the **same name**.
        Returns (count_created, status_message).
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return 0, "No stage open."

        markers = self._find_nav_markers(stage)
        if not markers:
            return 0, "No Nav_XXX / Nav.XXX prims found to convert."

        parent_path = self._ensure_nav_volumes_root(stage)
        mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        created = 0
        usd_context = omni.usd.get_context()

        for prim, _index in markers:
            try:
                marker_name = prim.GetName()
                world_xform = self._volume_xform_from_marker(prim, mpu)
                if world_xform is None:
                    carb.log_warn(f"[crowd.agents] Skip {prim.GetPath()} — no transform/bounds")
                    continue

                marker_path = prim.GetPath()
                # Remove placeholder cube first so it does not pollute the bake
                stage.RemovePrim(marker_path)

                omni.kit.commands.execute(
                    "CreateNavMeshVolumeCommand",
                    parent_prim_path=parent_path,
                    volume_type=NAVMESH_VOLUME_INCLUDE,
                    position=None,
                )
                selected = usd_context.get_selection().get_selected_prim_paths()
                if not selected:
                    carb.log_warn("[crowd.agents] CreateNavMeshVolumeCommand left no selection")
                    continue
                vol_path = Sdf.Path(selected[0])

                # Child of NavVolumes, keep the marker's name (Nav_001, etc.)
                target = parent_path.AppendChild(marker_name)
                target = Sdf.Path(
                    omni.usd.get_stage_next_free_path(stage, target.pathString, False)
                )
                try:
                    omni.kit.commands.execute(
                        "MovePrim",
                        path_from=vol_path,
                        path_to=target,
                    )
                    vol_path = target
                except Exception as move_exc:  # noqa: BLE001
                    carb.log_warn(
                        f"[crowd.agents] MovePrim to {target} failed ({move_exc}); "
                        "leaving volume at create path"
                    )

                local_xform = self._world_to_local_xform(stage, vol_path, world_xform)
                omni.kit.commands.execute(
                    "TransformPrim",
                    path=vol_path,
                    new_transform_matrix=local_xform,
                )

                created += 1
                carb.log_info(f"[crowd.agents] Nav marker → Include Volume {vol_path}")
            except Exception as exc:  # noqa: BLE001
                carb.log_error(f"[crowd.agents] Failed converting marker: {exc}")

        if created == 0:
            return 0, "Found Nav markers but failed to create Include Volumes."
        return (
            created,
            f"Created {created} Include Volume(s) under {parent_path} (names kept).",
        )

    @staticmethod
    def _ensure_nav_volumes_root(stage) -> Sdf.Path:
        """Ensure ``/World/NavVolumes`` exists as an Xform parent."""
        # Prefer an existing NavVolumes anywhere; else create under /World
        for prim in stage.Traverse():
            if prim.GetName() == "NavVolumes":
                return prim.GetPath()

        if not stage.GetPrimAtPath("/World"):
            UsdGeom.Xform.Define(stage, "/World")
        path = Sdf.Path("/World/NavVolumes")
        if not stage.GetPrimAtPath(path):
            UsdGeom.Xform.Define(stage, path)
            carb.log_info("[crowd.agents] Created /World/NavVolumes")
        return path

    @staticmethod
    def _world_to_local_xform(stage, prim_path: Sdf.Path, world_xform: Gf.Matrix4d) -> Gf.Matrix4d:
        """Convert a world matrix into local space for ``prim_path``'s parent."""
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return world_xform
        parent = prim.GetParent()
        if not parent or not parent.IsValid():
            return world_xform
        parent_xf = UsdGeom.Xformable(parent)
        if not parent_xf:
            return world_xform
        try:
            parent_world = parent_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return world_xform * parent_world.GetInverse()
        except Exception:  # noqa: BLE001
            return world_xform

    @staticmethod
    def _find_nav_markers(stage) -> List[Tuple[Usd.Prim, int]]:
        found: List[Tuple[Usd.Prim, int]] = []
        for prim in stage.Traverse():
            match = NAV_MARKER_PATTERN.match(prim.GetName())
            if not match:
                continue
            # Skip volumes already under NavVolumes (re-run safety)
            if "NavMeshVolume" in (prim.GetTypeName() or ""):
                continue
            found.append((prim, int(match.group(1))))
        found.sort(key=lambda item: item[1])
        return found

    @staticmethod
    def _marker_geom_and_bbox(
        prim: Usd.Prim,
    ) -> Tuple[Usd.Prim, Gf.Vec3d, Gf.Vec3d]:
        """Pick the boundable mesh (self or descendant) and its local AABB."""
        candidates = [prim]
        stack = list(prim.GetChildren())
        while stack:
            c = stack.pop()
            candidates.append(c)
            stack.extend(list(c.GetChildren()))

        best_prim = prim
        best = None
        best_vol = -1.0
        for cand in candidates:
            mn_mx = None
            attr = cand.GetAttribute("extent")
            if attr and attr.HasAuthoredValueOpinion():
                try:
                    ext = attr.Get()
                    if ext is not None and len(ext) >= 2:
                        mn_mx = (
                            Gf.Vec3d(float(ext[0][0]), float(ext[0][1]), float(ext[0][2])),
                            Gf.Vec3d(float(ext[1][0]), float(ext[1][1]), float(ext[1][2])),
                        )
                except Exception:  # noqa: BLE001
                    mn_mx = None
            if mn_mx is None:
                boundable = UsdGeom.Boundable(cand)
                if boundable:
                    try:
                        bbox = boundable.ComputeLocalBound(Usd.TimeCode.Default())
                        if hasattr(bbox, "ComputeAlignedRange"):
                            rng = bbox.ComputeAlignedRange()
                        elif hasattr(bbox, "GetRange"):
                            rng = bbox.GetRange()
                        else:
                            rng = bbox.GetBox()
                        mn, mx = rng.GetMin(), rng.GetMax()
                        mn_mx = (
                            Gf.Vec3d(float(mn[0]), float(mn[1]), float(mn[2])),
                            Gf.Vec3d(float(mx[0]), float(mx[1]), float(mx[2])),
                        )
                    except Exception:  # noqa: BLE001
                        mn_mx = None
            if mn_mx is None:
                continue
            mn, mx = mn_mx
            vol = abs((mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2]))
            if vol > best_vol:
                best_vol = vol
                best = mn_mx
                best_prim = cand

        if best is None:
            return prim, Gf.Vec3d(-1.0, -1.0, -1.0), Gf.Vec3d(1.0, 1.0, 1.0)
        return best_prim, best[0], best[1]

    @staticmethod
    def _volume_xform_from_marker(prim: Usd.Prim, mpu: float) -> Optional[Gf.Matrix4d]:
        """World matrix mapping the unit NavMesh volume onto the marker's box.

        NavMeshVolume local extent is a 1 m cube (half-extent 0.5 m → size 1/mpu
        in stage units). Scale that cube to the mesh local bbox, then apply the
        mesh local-to-world so the Include volume matches the replaced object.
        """
        geom, mn, mx = NavigationSystem._marker_geom_and_bbox(prim)
        xformable = UsdGeom.Xformable(geom)
        if not xformable:
            xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None
        marker_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        local_size = Gf.Vec3d(
            max(abs(float(mx[0] - mn[0])), 1e-4),
            max(abs(float(mx[1] - mn[1])), 1e-4),
            max(abs(float(mx[2] - mn[2])), 1e-4),
        )
        center_local = Gf.Vec3d(
            0.5 * (float(mn[0]) + float(mx[0])),
            0.5 * (float(mn[1]) + float(mx[1])),
            0.5 * (float(mn[2]) + float(mx[2])),
        )

        # p_world = p_volume_local * Scale(local_size*mpu) * T(center) * marker_world
        scale_m = Gf.Matrix4d(1.0)
        scale_m.SetScale(
            Gf.Vec3d(
                float(local_size[0]) * mpu,
                float(local_size[1]) * mpu,
                float(local_size[2]) * mpu,
            )
        )
        center_m = Gf.Matrix4d(1.0)
        center_m.SetTranslate(center_local)
        volume_world = scale_m * center_m * marker_world

        carb.log_info(
            f"[crowd.agents] Volume match {prim.GetName()} geom={geom.GetPath()}: "
            f"local_size=({local_size[0]:.3f},{local_size[1]:.3f},{local_size[2]:.3f}) "
            f"scale=({local_size[0]*mpu:.3f},{local_size[1]*mpu:.3f},{local_size[2]*mpu:.3f})"
        )
        return volume_world

    def snap_to_navmesh(self, position: Vec3) -> Optional[Vec3]:
        inav = self._acquire()
        if inav is None:
            return None
        navmesh = inav.get_navmesh()
        if navmesh is None:
            return None
        try:
            closest, _island = navmesh.query_closest_point(target=carb.Float3(*position))
            if closest is None:
                return None
            return (float(closest[0]), float(closest[1]), float(closest[2]))
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] query_closest_point failed: {exc}")
            return None

    def random_point(self, seed_key: str = "crowd.agents") -> Optional[Vec3]:
        inav = self._acquire()
        if inav is None:
            return None
        navmesh = inav.get_navmesh()
        if navmesh is None:
            return None
        self._random_counter += 1
        seed = 1000 + self._random_counter * 97
        try:
            try:
                inav.set_random_seed(seed_key, seed)
            except Exception:  # noqa: BLE001
                pass
            try:
                point = navmesh.query_random_point(seed_key)
            except TypeError:
                point = navmesh.query_random_point()
            if point is None:
                return None
            return (float(point[0]), float(point[1]), float(point[2]))
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] query_random_point failed: {exc}")
            return None

    def query_path(self, start: Vec3, end: Vec3) -> List[Vec3]:
        inav = self._acquire()
        if inav is None:
            return []
        navmesh = inav.get_navmesh()
        if navmesh is None:
            return []
        try:
            path = navmesh.query_shortest_path(
                start_pos=carb.Float3(*start),
                end_pos=carb.Float3(*end),
            )
            if path is None:
                return []
            points = path.get_points()
            if not points:
                return []
            return [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] query_shortest_path failed: {exc}")
            return []

    def set_destination(self, agent: "Agent", destination: Vec3) -> bool:
        start = self.snap_to_navmesh(agent.position) or agent.position
        end = self.snap_to_navmesh(destination) or destination
        waypoints = self.query_path(start, end)
        if len(waypoints) < 2:
            waypoints = [start, end]
            carb.log_warn("[crowd.agents] No NavMesh path — using direct line fallback")

        # Spread agents across the corridor instead of hugging the shortest-path edge
        if "lane_bias" not in agent.memory:
            agent.lane_bias = self._lane_rng.triangular(-1.0, 1.0, 0.0)
            agent.memory["lane_bias"] = agent.lane_bias
        else:
            agent.lane_bias = float(agent.memory.get("lane_bias", agent.lane_bias))
        waypoints = self._apply_lane_offsets(waypoints, agent)

        agent.position = start
        agent.destination = end
        agent.waypoints = waypoints
        agent.waypoint_index = 1 if len(waypoints) > 1 else 0
        agent.perception["at_destination"] = False
        agent.retry_timer = 0.0
        agent.last_move_pos = start
        agent.last_wp_index = agent.waypoint_index
        agent.last_waypoint_dist = _dist(
            start, waypoints[min(agent.waypoint_index, len(waypoints) - 1)]
        )
        carb.log_info(
            f"[crowd.agents] Path planned: {len(waypoints)} pts from {start} -> {end} "
            f"(lane_bias={agent.lane_bias:.2f})"
        )
        return True

    def apply_intent(self, agent: "Agent", intent: str, dt: float) -> None:
        if intent == "IDLE" or dt <= 0.0:
            return
        if intent not in ("GO_TO_SEAT", "EXIT", "WALK"):
            return
        if agent.perception.get("at_destination") and intent == "EXIT":
            # Already arrived at an exit — stay put. Seated agents must leave.
            if (agent.goal or "").startswith("Exit"):
                return
            agent.perception["at_destination"] = False
        if not agent.waypoints and agent.destination is not None:
            self.set_destination(agent, agent.destination)
        self._follow_waypoints(agent, dt)
        self._update_replan(agent, dt)

    def _follow_waypoints(self, agent: "Agent", dt: float) -> None:
        if not agent.waypoints:
            return

        # True seat/exit arrival — never mark seated on a mid-path node alone
        if agent.destination is not None:
            if _dist(agent.position, agent.destination) <= GOAL_ARRIVAL_EPSILON:
                agent.position = agent.destination
                agent.perception["at_destination"] = True
                agent.intent = "IDLE"
                agent.retry_timer = 0.0
                return

        # Skip checkpoints the agent has already passed (crowd shove / overshoot)
        self._advance_past_waypoints(agent)

        idx = min(agent.waypoint_index, len(agent.waypoints) - 1)
        target = agent.waypoints[idx]
        pos = agent.position
        dx, dy, dz = target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist <= WAYPOINT_EPSILON:
            if idx < len(agent.waypoints) - 1:
                agent.waypoint_index = idx + 1
            return

        step = min(agent.speed_mps * dt, dist)
        inv = step / dist if dist > 1e-8 else 0.0
        new_pos = (pos[0] + dx * inv, pos[1] + dy * inv, pos[2] + dz * inv)
        agent.position = new_pos
        agent.perception["at_destination"] = False
        # Face movement direction for FOV perception
        if abs(dx) + abs(dy) + abs(dz) > 1e-8:
            length = dist
            agent.facing = (dx / length, dy / length, dz / length)

    def _advance_past_waypoints(self, agent: "Agent") -> None:
        """Advance waypoint index when close to or past the current checkpoint.

        Prevents backtracking after crowding pushes an agent beyond a node.
        """
        wps = agent.waypoints
        if len(wps) < 2:
            return
        pos = agent.position
        idx = agent.waypoint_index
        # Never skip the final destination node via overshoot logic alone
        while idx < len(wps) - 1:
            curr = wps[idx]
            nxt = wps[idx + 1]
            d_curr = _dist(pos, curr)
            if d_curr <= WAYPOINT_EPSILON:
                idx += 1
                continue

            # Past the checkpoint if we're on the far side toward the next node
            cx, cy, cz = curr
            nx, ny, nz = nxt[0] - cx, nxt[1] - cy, nxt[2] - cz
            ax, ay, az = pos[0] - cx, pos[1] - cy, pos[2] - cz
            seg_len_sq = nx * nx + ny * ny + nz * nz
            if seg_len_sq > 1e-8:
                t = (ax * nx + ay * ny + az * nz) / seg_len_sq
                if t > 0.0:
                    idx += 1
                    continue

            # Or simply closer to the next checkpoint than this one
            if _dist(pos, nxt) + 0.05 < d_curr:
                idx += 1
                continue
            break

        agent.waypoint_index = idx

    def _update_replan(self, agent: "Agent", dt: float) -> None:
        """Replan ~1/s when not closing on the current waypoint fast enough."""
        if agent.perception.get("at_destination"):
            agent.retry_timer = 0.0
            agent.last_move_pos = agent.position
            return
        if agent.destination is None or not agent.waypoints or dt <= 0.0:
            return

        idx = min(agent.waypoint_index, len(agent.waypoints) - 1)
        target = agent.waypoints[idx]
        wp_dist = _dist(agent.position, target)

        # New checkpoint → reset progress tracking (don't treat index jumps as stall)
        if int(getattr(agent, "last_wp_index", -1)) != idx:
            agent.last_wp_index = idx
            agent.last_waypoint_dist = wp_dist
            agent.retry_timer = 0.0
            agent.last_move_pos = agent.position
            return

        dt_safe = min(dt, REPLAN_DT_CAP_S)
        prev = float(getattr(agent, "last_waypoint_dist", wp_dist))
        closing_mps = (prev - wp_dist) / dt_safe
        agent.last_waypoint_dist = wp_dist
        agent.last_move_pos = agent.position

        if closing_mps >= MIN_WP_CLOSING_MPS:
            agent.retry_timer = 0.0
            return

        agent.retry_timer += dt_safe
        if agent.retry_timer < RETRY_TIMEOUT_S:
            return

        agent.retry_timer = 0.0
        bias = float(getattr(agent, "lane_bias", 0.0) or 0.0)
        agent.lane_bias = max(-1.0, min(1.0, bias + self._lane_rng.uniform(-0.35, 0.35)))
        agent.memory["lane_bias"] = agent.lane_bias
        carb.log_info(
            f"[crowd.agents] Replan {agent.agent_id} "
            f"(wp closing={closing_mps:.2f} m/s < {MIN_WP_CLOSING_MPS})"
        )
        self.set_destination(agent, agent.destination)

    def _bake_agent_radius(self) -> float:
        """NavMesh bake agent radius — safe lower-bound clearance on the walkable mesh."""
        inav = self._acquire()
        if inav is None:
            return DEFAULT_BAKE_RADIUS
        navmesh = inav.get_navmesh()
        if navmesh is None:
            return DEFAULT_BAKE_RADIUS
        try:
            # Prefer min radius (tightest bake); fall back to max / default
            if hasattr(navmesh, "get_agent_min_radius"):
                r = float(navmesh.get_agent_min_radius())
                if r > 1e-4:
                    return r
            if hasattr(navmesh, "get_agent_max_radius"):
                r = float(navmesh.get_agent_max_radius())
                if r > 1e-4:
                    return r
        except Exception:  # noqa: BLE001
            pass
        return DEFAULT_BAKE_RADIUS

    def _apply_lane_offsets(self, waypoints: List[Vec3], agent: "Agent") -> List[Vec3]:
        """Cheap lateral jitter on intermediate waypoints, capped by bake agent radius.

        The NavMesh is already eroded by the bake radius, so offsets within that
        radius stay on the walkable surface without corridor probing.
        """
        if len(waypoints) < 3:
            return waypoints

        up = _stage_up_axis()
        bias = float(getattr(agent, "lane_bias", 0.0) or 0.0)
        max_offset = max(0.05, self._bake_agent_radius() * LANE_RADIUS_SCALE)
        offset = bias * max_offset

        out: List[Vec3] = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            prev_pt, curr, next_pt = waypoints[i - 1], waypoints[i], waypoints[i + 1]
            tangent = _horiz_dir(prev_pt, next_pt, up) or _horiz_dir(prev_pt, curr, up)
            if tangent is None or abs(offset) < 1e-4:
                out.append(curr)
                continue

            left = (-tangent[1], tangent[0])
            shifted = _offset_horizontal(curr, left, offset, up)
            snapped = self.snap_to_navmesh(shifted)
            out.append(snapped if snapped is not None else curr)

        out.append(waypoints[-1])
        return out


def _stage_up_axis() -> str:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return "Y"
    meta = stage.GetMetadata("upAxis")
    return str(meta) if meta is not None else "Y"


def _horiz_dir(a: Vec3, b: Vec3, up: str) -> Optional[Tuple[float, float]]:
    if up == "Y":
        dx, dz = b[0] - a[0], b[2] - a[2]
    else:
        dx, dz = b[0] - a[0], b[1] - a[1]
    length = math.sqrt(dx * dx + dz * dz)
    if length < 1e-8:
        return None
    return (dx / length, dz / length)


def _offset_horizontal(
    origin: Vec3, direction: Tuple[float, float], distance: float, up: str
) -> Vec3:
    if up == "Y":
        return (
            origin[0] + direction[0] * distance,
            origin[1],
            origin[2] + direction[1] * distance,
        )
    return (
        origin[0] + direction[0] * distance,
        origin[1] + direction[1] * distance,
        origin[2],
    )


def _dist(a: Vec3, b: Vec3) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
