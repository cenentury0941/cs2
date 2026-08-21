# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""NavMesh baking and path following via omni.anim.navigation.core."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import carb
import omni.kit.commands
import omni.usd
from pxr import Sdf

if TYPE_CHECKING:
    from .agent import Agent

Vec3 = Tuple[float, float, float]
ARRIVAL_EPSILON = 0.15


class NavigationSystem:
    """Wraps Omniverse NavMesh bake + shortest-path queries."""

    def __init__(self) -> None:
        self._interface = None

    def _acquire(self):
        if self._interface is not None:
            return self._interface
        try:
            import omni.anim.navigation.core as nav

            self._interface = nav.acquire_interface()
            return self._interface
        except Exception as exc:  # noqa: BLE001 — surface as status to UI
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
        """Create a NavMesh Include Volume under /World if the stage has none."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if type_name and "NavMeshVolume" in type_name:
                return True
            # Schema-backed volumes may appear as Xform with nav attrs
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
        """Bake (or re-bake) the stage NavMesh. Requires an Include Volume."""
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
        try:
            try:
                inav.set_random_seed(seed_key, 42)
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
        """Plan a path from the agent to destination and store waypoints."""
        start = self.snap_to_navmesh(agent.position) or agent.position
        end = self.snap_to_navmesh(destination) or destination
        waypoints = self.query_path(start, end)
        if len(waypoints) < 2:
            # Fallback: direct line if bake/query fails but we still want motion
            waypoints = [start, end]
            carb.log_warn(
                "[crowd.agents] No NavMesh path — using direct line fallback"
            )
        agent.position = start
        agent.destination = end
        agent.waypoints = waypoints
        agent.waypoint_index = 1 if len(waypoints) > 1 else 0
        agent.perception["at_destination"] = False
        return True

    def apply_intent(self, agent: "Agent", intent: str, dt: float) -> None:
        if intent == "IDLE" or dt <= 0.0:
            return

        if intent in ("GO_TO_SEAT", "EXIT", "WALK"):
            if not agent.waypoints and agent.destination is not None:
                self.set_destination(agent, agent.destination)
            self._follow_waypoints(agent, dt)

    def _follow_waypoints(self, agent: "Agent", dt: float) -> None:
        if not agent.waypoints:
            return

        idx = min(agent.waypoint_index, len(agent.waypoints) - 1)
        target = agent.waypoints[idx]
        pos = agent.position
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dz = target[2] - pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist <= ARRIVAL_EPSILON:
            agent.position = target
            if idx >= len(agent.waypoints) - 1:
                agent.perception["at_destination"] = True
                agent.intent = "IDLE"
                return
            agent.waypoint_index = idx + 1
            return

        step = min(agent.speed_mps * dt, dist)
        inv = step / dist
        agent.position = (pos[0] + dx * inv, pos[1] + dy * inv, pos[2] + dz * inv)
        agent.perception["at_destination"] = False
