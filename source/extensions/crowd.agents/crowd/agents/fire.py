# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Fire hazard: click-to-place cuboid + range/FOV perception."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import carb
import omni.usd
from pxr import Gf, UsdGeom

if TYPE_CHECKING:
    from .agent import Agent

Vec3 = Tuple[float, float, float]

FIRE_PREVIEW_PATH = "/World/CrowdAgents/FirePreview"
FIRE_PATH = "/World/CrowdAgents/Fire"
# Roughly agent footprint / torso scale (capsule radius 0.35, height 1.7)
FIRE_CUBE_SIZE = 0.8
# Perception: fire volume ≈ cube as a sphere; agent must be in range and FOV cone
FIRE_VOLUME_RADIUS_M = FIRE_CUBE_SIZE * 0.5
FIRE_VIEW_RANGE_M = 20.0
FIRE_FOV_DEG = 210.0  # total horizontal field of view
# Line-of-sight: sync mesh BVH ray eye → fire; retry when blocked
LOS_RETRY_S = 0.35
LOS_HIT_SLACK_M = 0.35  # treat hits near the fire as "reached fire"
EYE_HEIGHT_FRAC = 0.9
LOS_DEBUG_LOGS = 8  # log first N LOS decisions to console
# Skip non-occluders (nav volumes, etc.) and re-cast; never treat them as clear
LOS_SKIP_MAX = 8
# Require consecutive clear casts before sticky sighting (filters one-frame BVH gaps)
LOS_CONFIRM_CLEARS = 2

OnPlacedFn = Callable[[Vec3], None]


def fire_in_view(
    agent_pos: Vec3,
    facing: Vec3,
    fire_pos: Vec3,
    *,
    up: str = "Y",
    view_range_m: float = FIRE_VIEW_RANGE_M,
    volume_radius_m: float = FIRE_VOLUME_RADIUS_M,
    fov_deg: float = FIRE_FOV_DEG,
) -> bool:
    """True if fire volume overlaps the agent's forward view cone + range."""
    to_fire = _horizontal_delta(agent_pos, fire_pos, up)
    if to_fire is None:
        return True

    dist = math.sqrt(to_fire[0] * to_fire[0] + to_fire[1] * to_fire[1])
    if dist > view_range_m + volume_radius_m:
        return False

    face = _horizontal_facing(facing, up)
    if face is None:
        return dist <= view_range_m + volume_radius_m

    inv = 1.0 / dist
    dir_x, dir_z = to_fire[0] * inv, to_fire[1] * inv
    dot = max(-1.0, min(1.0, face[0] * dir_x + face[1] * dir_z))
    half_fov = math.radians(fov_deg * 0.5)
    return math.acos(dot) <= half_fov


def _horizontal_delta(a: Vec3, b: Vec3, up: str) -> Optional[Tuple[float, float]]:
    if up == "Z":
        dx, dy = b[0] - a[0], b[1] - a[1]
    else:
        dx, dy = b[0] - a[0], b[2] - a[2]
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None
    return (dx, dy)


def _horizontal_facing(facing: Vec3, up: str) -> Optional[Tuple[float, float]]:
    if up == "Z":
        fx, fy = facing[0], facing[1]
    else:
        fx, fy = facing[0], facing[2]
    length = math.sqrt(fx * fx + fy * fy)
    if length < 1e-8:
        return None
    return (fx / length, fy / length)


class FireSystem:
    """Place a red fire cuboid; agents see it via range + FOV + wall LOS."""

    def __init__(self) -> None:
        self._active = False
        self._position: Optional[Vec3] = None
        self._placing = False
        self._on_placed: Optional[OnPlacedFn] = None
        self._scene_view = None
        self._viewport_frame = None
        self._selection_guard = None
        self._preview_query_id = 0
        self._confirm_query_id = 0
        self._preview_visible = False
        self._los_time = 0.0
        self._los_clear: Dict[str, bool] = {}
        self._los_clear_streak: Dict[str, int] = {}
        self._los_next_check: Dict[str, float] = {}
        self._mesh_raycast = None
        self._mesh_ctx = None
        self._mesh_raycast_failed = False
        self._los_debug_left = LOS_DEBUG_LOGS
        self._bvh_has_geometry = False

    def tick(self, dt: float) -> None:
        """Advance LOS timers (call once per sim update)."""
        if dt > 0.0:
            self._los_time += dt

    def _clear_los_state(self) -> None:
        self._los_clear.clear()
        self._los_clear_streak.clear()
        self._los_next_check.clear()
        self._los_debug_left = LOS_DEBUG_LOGS
        # Keep _bvh_has_geometry — stage meshes persist across fire reset

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_placing(self) -> bool:
        return self._placing

    @property
    def position(self) -> Optional[Vec3]:
        return self._position

    def begin_placement(self, on_placed: OnPlacedFn) -> bool:
        """Enter click-to-place mode. Returns False if viewport overlay failed."""
        if self._placing:
            self.cancel_placement()
        else:
            self._detach_viewport_overlay()
            self._hide_preview()
        self._on_placed = on_placed
        if not self._attach_viewport_overlay():
            self._on_placed = None
            return False
        self._placing = True
        carb.log_info("[crowd.agents] Fire placement mode — click viewport to spawn")
        return True

    def cancel_placement(self) -> None:
        was = self._placing
        self._placing = False
        self._on_placed = None
        self._detach_viewport_overlay()
        self._hide_preview()
        if was:
            carb.log_info("[crowd.agents] Fire placement cancelled")

    def trigger_at(self, position: Vec3) -> None:
        """Spawn / move the fire cuboid and mark fire active."""
        self.cancel_placement()
        self._active = True
        self._position = position
        self._clear_los_state()
        self._ensure_cube(FIRE_PATH, position, preview=False)
        self._hide_preview()
        carb.log_info(
            f"[crowd.agents] Fire active at "
            f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        )

    def reset(self) -> None:
        self.cancel_placement()
        self._active = False
        self._position = None
        self._clear_los_state()
        self._remove_prim(FIRE_PATH)
        self._remove_prim(FIRE_PREVIEW_PATH)

    def is_visible_to(self, agent: "Agent", up: str = "Y") -> bool:
        """Range + FOV + mesh LOS. Sticky after first confirmed sighting."""
        if not self._active or self._position is None:
            return False
        if agent.memory.get("fire_seen"):
            return True

        facing = getattr(agent, "facing", None) or (0.0, 0.0, 1.0)
        if not fire_in_view(agent.position, facing, self._position, up=up):
            return False

        aid = agent.agent_id
        if self._los_clear.get(aid):
            agent.memory["fire_seen"] = True
            return True

        # In FOV+range — require a clear mesh ray before granting vision
        self._update_los(agent, up)
        if self._los_clear.get(aid):
            agent.memory["fire_seen"] = True
            return True
        return False

    def _update_los(self, agent: "Agent", up: str) -> None:
        aid = agent.agent_id
        if self._los_time < self._los_next_check.get(aid, 0.0):
            return
        if self._position is None:
            return

        eye = _lift_point(agent.position, agent.height * EYE_HEIGHT_FRAC, up)
        # Aim slightly above fire center so the ray stays off the floor mesh
        fire_pt = _lift_point(self._position, max(FIRE_CUBE_SIZE * 0.5, 1.2), up)
        dx = fire_pt[0] - eye[0]
        dy = fire_pt[1] - eye[1]
        dz = fire_pt[2] - eye[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-4:
            self._los_clear[aid] = True
            return

        inv = 1.0 / dist
        direction = (dx * inv, dy * inv, dz * inv)
        # End just short of the fire so a fire-cube hit isn't required for "clear"
        max_t = max(0.05, dist - LOS_HIT_SLACK_M)

        clear, hit_path, hit_dist = self._mesh_los_test(eye, direction, max_t)
        if clear:
            streak = self._los_clear_streak.get(aid, 0) + 1
            self._los_clear_streak[aid] = streak
            # Sticky only after consecutive confirms — one BVH miss must not grant vision
            self._los_clear[aid] = streak >= LOS_CONFIRM_CLEARS
        else:
            self._los_clear_streak[aid] = 0
            self._los_clear[aid] = False
        self._los_next_check[aid] = self._los_time + LOS_RETRY_S

        if self._los_debug_left > 0:
            self._los_debug_left -= 1
            carb.log_info(
                f"[crowd.agents] LOS {aid}: clear={clear} "
                f"streak={self._los_clear_streak.get(aid, 0)}/{LOS_CONFIRM_CLEARS} "
                f"dist={dist:.2f} hit_dist={hit_dist} hit={hit_path or '(none)'}"
            )

    def _acquire_mesh_raycast(self):
        if self._mesh_raycast_failed:
            return None, None
        if self._mesh_raycast is not None and self._mesh_ctx is not None:
            return self._mesh_raycast, self._mesh_ctx
        try:
            import omni.kit.mesh.raycast as mesh_raycast

            iface = mesh_raycast.acquire_mesh_raycast_interface()
            ctx = iface.create_context()
            try:
                iface.set_bvh_refresh_rate(mesh_raycast.BvhRefreshRate.FAST, True)
            except Exception:  # noqa: BLE001
                pass
            self._mesh_raycast = iface
            self._mesh_ctx = ctx
            carb.log_info("[crowd.agents] Mesh raycast LOS ready")
            return iface, ctx
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] Mesh raycast unavailable: {exc}")
            self._mesh_raycast_failed = True
            return None, None

    def _collect_disallow_paths(self) -> list:
        """Meshes that must never occlude (or grant false clear): fire, agents, nav vols."""
        disallow = [FIRE_PATH, FIRE_PREVIEW_PATH, "/World/CrowdAgents/AgentFovViz"]
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return disallow

        root = stage.GetPrimAtPath("/World/CrowdAgents")
        if root and root.IsValid():
            for child in root.GetChildren():
                name = child.GetName()
                if name.startswith("Agent_"):
                    disallow.append(str(child.GetPath()))

        # Nav include volumes are large boxes; hitting them must not count as clear LOS
        for nav_path in ("/World/NavVolumes",):
            nav_root = stage.GetPrimAtPath(nav_path)
            if not nav_root or not nav_root.IsValid():
                continue
            stack = [nav_root]
            while stack:
                prim = stack.pop()
                disallow.append(str(prim.GetPath()))
                stack.extend(list(prim.GetChildren()))

        return disallow

    def _closest_raycast(self, iface, ctx, origin: Vec3, direction: Vec3, max_t: float):
        """Try common vector types; return (result, error)."""
        last_err = None
        for o, d in (
            (origin, direction),
            (Gf.Vec3d(*origin), Gf.Vec3d(*direction)),
            (carb.Float3(*origin), carb.Float3(*direction)),
        ):
            try:
                return iface.closestRaycast(o, d, float(max_t), ctx), None
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        return None, last_err

    def _probe_bvh_geometry(self, iface, ctx) -> None:
        """One downward cast so empty eye→fire rays are not trusted before BVH exists."""
        if self._bvh_has_geometry:
            return
        stage = omni.usd.get_context().get_stage()
        up = "Y"
        if stage is not None:
            meta = stage.GetMetadata("upAxis")
            if meta is not None:
                up = str(meta)
        base = self._position or (0.0, 0.0, 0.0)
        origin = _lift_point(base, 8.0, up)
        direction = (0.0, -1.0, 0.0) if up != "Z" else (0.0, 0.0, -1.0)
        result, _err = self._closest_raycast(iface, ctx, origin, direction, 20.0)
        if result is None:
            return
        if int(getattr(result, "meshIndex", -1)) >= 0:
            self._bvh_has_geometry = True

    def _mesh_los_test(
        self, origin: Vec3, direction: Vec3, max_t: float
    ) -> Tuple[bool, str, Optional[float]]:
        """Return (clear, hit_path, hit_distance). Fail-closed: blocked if unsure.

        Non-occluding hits (nav volumes, agents, fire) are skipped and the ray is
        re-cast — they must never grant vision through a wall behind them.
        """
        iface, ctx = self._acquire_mesh_raycast()
        if iface is None or ctx is None:
            # Do NOT fall back to FOV-only — that is what caused see-through walls
            return False, "(no-raycast)", None

        try:
            disallow = self._collect_disallow_paths()
        except Exception:  # noqa: BLE001
            disallow = [FIRE_PATH, FIRE_PREVIEW_PATH, "/World/CrowdAgents/AgentFovViz"]

        try:
            if hasattr(iface, "set_disallowed_mesh_paths"):
                iface.set_disallowed_mesh_paths(ctx, disallow)
        except Exception:  # noqa: BLE001
            pass

        self._probe_bvh_geometry(iface, ctx)

        for _attempt in range(LOS_SKIP_MAX):
            try:
                if hasattr(iface, "set_disallowed_mesh_paths"):
                    iface.set_disallowed_mesh_paths(ctx, disallow)
            except Exception:  # noqa: BLE001
                pass

            result, last_err = self._closest_raycast(iface, ctx, origin, direction, max_t)
            if result is None:
                if last_err is not None:
                    carb.log_warn(f"[crowd.agents] closestRaycast failed: {last_err}")
                return False, "(raycast-error)", None

            mesh_index = int(getattr(result, "meshIndex", -1))
            if mesh_index < 0:
                # Empty segment. Until BVH has returned geometry, treat as blocked
                # (cold BVH / bvhBuildOnFirstRequired would otherwise fail-open).
                if not self._bvh_has_geometry:
                    return False, "(bvh-cold)", None
                return True, "", None

            self._bvh_has_geometry = True

            path = ""
            try:
                raw = iface.get_mesh_path_from_index(mesh_index)
                path = str(raw) if raw is not None else ""
            except Exception:  # noqa: BLE001
                path = ""

            hit_dist = None
            for attr in ("t", "distance", "hit_t", "dist"):
                if hasattr(result, attr):
                    try:
                        hit_dist = float(getattr(result, attr))
                        break
                    except Exception:  # noqa: BLE001
                        pass

            # Empty / unknown path still blocks — something was hit
            if not path:
                return False, "(unknown)", hit_dist

            if _path_is_fire(path) or not _path_blocks_los(path):
                # Skip this mesh and keep looking for a real occluder behind it
                if path not in disallow:
                    disallow.append(path)
                continue

            if hit_dist is not None and hit_dist >= max_t - 1e-3:
                # Hit at the end of the segment — treat as reaching the fire area
                return True, path, hit_dist

            return False, path, hit_dist

        # Exhausted skip budget while only hitting non-blockers — fail closed
        return False, "(skip-budget)", None

    def _attach_viewport_overlay(self) -> bool:
        try:
            import omni.ui.scene as sc
            from omni.kit.viewport.utility import (
                disable_selection,
                get_active_viewport,
                get_active_viewport_window,
            )
        except ImportError as exc:
            carb.log_error(f"[crowd.agents] Viewport APIs unavailable: {exc}")
            return False

        vp_window = get_active_viewport_window()
        viewport_api = get_active_viewport()
        if vp_window is None or viewport_api is None:
            carb.log_warn("[crowd.agents] No active viewport for fire placement")
            return False

        try:
            self._selection_guard = disable_selection(viewport_api)
        except Exception:  # noqa: BLE001
            self._selection_guard = None

        try:
            self._viewport_frame = vp_window.get_frame("crowd.agents.fire_place")
            with self._viewport_frame:
                self._scene_view = sc.SceneView(
                    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT
                )
                viewport_api.add_scene_view(self._scene_view)
                with self._scene_view.scene:
                    sc.Screen(gesture=_make_fire_gestures(self, viewport_api))
        except Exception as exc:  # noqa: BLE001
            carb.log_error(f"[crowd.agents] Failed to attach fire overlay: {exc}")
            self._detach_viewport_overlay()
            return False
        return True

    def _detach_viewport_overlay(self) -> None:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport_api = get_active_viewport()
            if viewport_api is not None and self._scene_view is not None:
                viewport_api.remove_scene_view(self._scene_view)
        except Exception:  # noqa: BLE001
            pass
        self._scene_view = None
        if self._viewport_frame is not None:
            try:
                self._viewport_frame.clear()
            except Exception:  # noqa: BLE001
                pass
        self._viewport_frame = None
        self._selection_guard = None

    def _query_world(
        self,
        viewport_api,
        mouse_ndc,
        on_hit: Callable[[Vec3], None],
        *,
        confirm: bool,
    ) -> None:
        try:
            pixel, valid = viewport_api.map_ndc_to_texture_pixel(mouse_ndc)
        except Exception:  # noqa: BLE001
            return
        if not valid:
            return

        if confirm:
            self._confirm_query_id += 1
            query_id = self._confirm_query_id
        else:
            self._preview_query_id += 1
            query_id = self._preview_query_id

        def _on_query(path, pos, *_args) -> None:
            if confirm:
                if query_id != self._confirm_query_id:
                    return
            else:
                if query_id != self._preview_query_id or not self._placing:
                    return
            if not path or pos is None:
                return
            try:
                world = (float(pos[0]), float(pos[1]), float(pos[2]))
            except Exception:  # noqa: BLE001
                return
            on_hit(world)

        try:
            viewport_api.request_query(pixel, _on_query)
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] Viewport query failed: {exc}")

    def _on_hover_ndc(self, viewport_api, mouse_ndc) -> None:
        if not self._placing:
            return
        self._query_world(viewport_api, mouse_ndc, self._set_preview, confirm=False)

    def _on_click_ndc(self, viewport_api, mouse_ndc) -> None:
        if not self._placing:
            return

        def _confirm(world: Vec3) -> None:
            if not self._placing:
                return
            cb = self._on_placed
            if cb is None:
                return
            cb(world)

        self._query_world(viewport_api, mouse_ndc, _confirm, confirm=True)

    def _set_preview(self, position: Vec3) -> None:
        self._ensure_cube(FIRE_PREVIEW_PATH, position, preview=True)
        self._preview_visible = True

    def _hide_preview(self) -> None:
        self._preview_visible = False
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(FIRE_PREVIEW_PATH)
        if prim and prim.IsValid():
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
            except Exception:  # noqa: BLE001
                self._remove_prim(FIRE_PREVIEW_PATH)

    def _ensure_cube(self, path: str, floor_pos: Vec3, *, preview: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        self._ensure_xform(stage, "/World")
        self._ensure_xform(stage, "/World/CrowdAgents")

        size = FIRE_CUBE_SIZE
        center = self._lift(floor_pos, size * 0.5)
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.GetTypeName() == "Cube":
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
            translate_op.Set(Gf.Vec3d(*center))
            UsdGeom.Imageable(prim).MakeVisible()
            return

        if prim and prim.IsValid():
            stage.RemovePrim(path)

        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(size)
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*center))

        color = Gf.Vec3f(1.0, 0.35, 0.1) if preview else Gf.Vec3f(0.95, 0.05, 0.02)
        gprim = UsdGeom.Gprim(cube.GetPrim())
        gprim.CreateDisplayColorAttr([color])
        try:
            opacity = 0.45 if preview else 1.0
            gprim.CreateDisplayOpacityAttr([opacity])
        except Exception:  # noqa: BLE001
            pass
        UsdGeom.Imageable(cube.GetPrim()).MakeVisible()

    @staticmethod
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

    @staticmethod
    def _ensure_xform(stage, path: str) -> None:
        if not stage.GetPrimAtPath(path):
            UsdGeom.Xform.Define(stage, path)

    @staticmethod
    def _remove_prim(path: str) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)


def _path_is_fire(path: str) -> bool:
    p = path.replace("\\", "/")
    return p == FIRE_PATH or p.startswith(FIRE_PATH + "/") or p.endswith("/Fire")


def _path_blocks_los(path: str) -> bool:
    """Walls/seats/venue mesh block; agents, fire preview, nav volumes do not.

    Empty/unknown paths block (fail-closed) — a hit with no path still occludes.
    """
    if not path:
        return True
    p = path.replace("\\", "/")
    lower = p.lower()
    if _path_is_fire(p):
        return False
    if p == FIRE_PREVIEW_PATH or p.startswith(FIRE_PREVIEW_PATH):
        return False
    if "/CrowdAgents/Agent_" in p or p.startswith("/World/CrowdAgents/Agent_"):
        return False
    if "/CrowdAgents/AgentFovViz" in p or p.endswith("/AgentFovViz"):
        return False
    if "navmesh" in lower or "/navvolumes" in lower or "navmeshvolume" in lower:
        return False
    return True


def _lift_point(floor_point: Vec3, amount: float, up: str) -> Vec3:
    x, y, z = floor_point
    if up == "Z":
        return (x, y, z + amount)
    return (x, y + amount, z)


def _make_fire_gestures(fire: FireSystem, viewport_api):
    """Build viewport Screen gestures (imported lazily so tests need no UI scene)."""
    import omni.ui.scene as sc

    class FireHoverGesture(sc.HoverGesture):
        def on_changed(self):  # noqa: N802 — Kit gesture API
            fire._on_hover_ndc(viewport_api, self.sender.gesture_payload.mouse)

    class FireClickGesture(sc.ClickGesture):
        def on_ended(self):  # noqa: N802 — Kit gesture API
            fire._on_click_ndc(viewport_api, self.sender.gesture_payload.mouse)

    return [FireHoverGesture(), FireClickGesture()]
