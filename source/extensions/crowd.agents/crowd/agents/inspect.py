# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Viewport click/hover → inspect agent via camera-ray capsule tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import carb

if TYPE_CHECKING:
    from .agent import Agent
    from .simulation import Simulation

Vec3 = Tuple[float, float, float]

# Inflate capsule radius slightly so hover/click is forgiving
_PICK_RADIUS_SCALE = 1.35
_MAX_RAY_T = 500.0


class AgentInspectPicker:
    """Screen overlay: pick the frontmost agent under the cursor (ray vs capsule)."""

    def __init__(self, simulation: "Simulation") -> None:
        self._simulation = simulation
        self._scene_view = None
        self._viewport_frame = None
        self._attached = False
        self._attach_failed_logged = False

    def ensure_attached(self) -> None:
        if self._attached:
            return
        if self._attach():
            self._attached = True
            self._attach_failed_logged = False

    def destroy(self) -> None:
        self._detach()
        self._attached = False

    def _attach(self) -> bool:
        try:
            import omni.ui.scene as sc
            from omni.kit.viewport.utility import (
                get_active_viewport,
                get_active_viewport_window,
            )
        except ImportError as exc:
            if not self._attach_failed_logged:
                carb.log_warn(f"[crowd.agents] Agent inspect picker unavailable: {exc}")
                self._attach_failed_logged = True
            return False

        vp_window = get_active_viewport_window()
        viewport_api = get_active_viewport()
        if vp_window is None or viewport_api is None:
            return False

        try:
            self._viewport_frame = vp_window.get_frame("crowd.agents.agent_inspect")
            with self._viewport_frame:
                # Do not use PRESERVE_ASPECT_FIT — letterboxing skewed NDC vs the
                # rendered image and made picks sit up/left of the cursor.
                self._scene_view = sc.SceneView()
                viewport_api.add_scene_view(self._scene_view)
                with self._scene_view.scene:
                    sc.Screen(gesture=_make_inspect_gestures(self))
        except Exception as exc:  # noqa: BLE001
            if not self._attach_failed_logged:
                carb.log_warn(
                    f"[crowd.agents] Failed to attach agent inspect overlay: {exc}"
                )
                self._attach_failed_logged = True
            self._detach()
            return False
        carb.log_info("[crowd.agents] Agent inspect ray-pick overlay ready")
        return True

    def _detach(self) -> None:
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

    def on_click_ndc(self, mouse_ndc) -> None:
        if self._simulation.is_placing_fire:
            return
        if not self._simulation.has_agents:
            return
        agent = self._pick_agent_at_ndc(mouse_ndc)
        if agent is not None:
            self._simulation.select_agent(agent.agent_id)
        else:
            self._simulation.select_agent(None)

    def on_hover_ndc(self, mouse_ndc) -> None:
        if self._simulation.is_placing_fire or not self._simulation.has_agents:
            self._simulation.set_hovered_agent(None)
            return
        agent = self._pick_agent_at_ndc(mouse_ndc)
        self._simulation.set_hovered_agent(agent.agent_id if agent else None)

    def on_hover_ended(self) -> None:
        self._simulation.set_hovered_agent(None)

    def _pick_agent_at_ndc(self, mouse_ndc) -> Optional["Agent"]:
        ray = self._ray_from_ndc(mouse_ndc)
        if ray is None:
            return None
        origin, direction = ray
        return self._simulation.pick_frontmost_agent(origin, direction)

    def _ray_from_ndc(self, mouse_ndc) -> Optional[Tuple[Vec3, Vec3]]:
        """World ray under the cursor using the SceneView camera (matches gesture NDC)."""
        if self._scene_view is None:
            return None
        try:
            import omni.ui.scene as sc

            x, y = _as_vec2(mouse_ndc)
            # Kit binding requires omni.ui.scene.Vector2 — plain tuples are rejected.
            origin, direction = self._scene_view.get_ray_from_ndc(sc.Vector2(x, y))
            o = _as_vec3(origin)
            d = _as_vec3(direction)
            if o is None or d is None:
                return None
            length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            if length < 1e-8:
                return None
            inv = 1.0 / length
            return o, (d[0] * inv, d[1] * inv, d[2] * inv)
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[crowd.agents] get_ray_from_ndc failed: {exc}")
            return None


def _make_inspect_gestures(picker: AgentInspectPicker):
    import omni.ui.scene as sc

    class InspectHoverGesture(sc.HoverGesture):
        def on_changed(self):  # noqa: N802 — Kit gesture API
            picker.on_hover_ndc(self.sender.gesture_payload.mouse)

        def on_ended(self):  # noqa: N802 — Kit gesture API
            picker.on_hover_ended()

    class InspectClickGesture(sc.ClickGesture):
        def on_ended(self):  # noqa: N802 — Kit gesture API
            picker.on_click_ndc(self.sender.gesture_payload.mouse)

    return [InspectHoverGesture(), InspectClickGesture()]


def ray_capsule_intersection_t(
    origin: Vec3,
    direction: Vec3,
    feet: Vec3,
    height: float,
    radius: float,
    *,
    up: str = "Y",
    max_t: float = _MAX_RAY_T,
) -> Optional[float]:
    """Smallest t > 0 where ray hits a vertical capsule, or None."""
    r = max(radius, 1e-4)
    # Capsule axis: sphere centers at feet+r and feet+height-r along up
    axis_len = max(height - 2.0 * r, 0.0)
    a0 = _lift(feet, r, up)
    a1 = _lift(feet, r + axis_len, up)

    best: Optional[float] = None

    def _consider(t: Optional[float]) -> None:
        nonlocal best
        if t is None or t <= 1e-5 or t > max_t:
            return
        if best is None or t < best:
            best = t

    _consider(_ray_sphere_t(origin, direction, a0, r, max_t=max_t))
    _consider(_ray_sphere_t(origin, direction, a1, r, max_t=max_t))
    _consider(_ray_finite_cylinder_t(origin, direction, a0, a1, r, up, max_t=max_t))
    return best


def pick_frontmost_along_ray(
    agents: Sequence["Agent"],
    origin: Vec3,
    direction: Vec3,
    *,
    up: str = "Y",
    radius_scale: float = _PICK_RADIUS_SCALE,
) -> Optional["Agent"]:
    """Return the agent whose capsule is hit first along the view ray."""
    best_agent: Optional["Agent"] = None
    best_t: Optional[float] = None
    for agent in agents:
        t = ray_capsule_intersection_t(
            origin,
            direction,
            agent.position,
            agent.height,
            agent.radius * radius_scale,
            up=up,
        )
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_agent = agent
    return best_agent


def horizontal_distance(a: Vec3, b: Vec3, up: str) -> float:
    if up == "Z":
        dx, dy = b[0] - a[0], b[1] - a[1]
        return math.sqrt(dx * dx + dy * dy)
    dx, dz = b[0] - a[0], b[2] - a[2]
    return math.sqrt(dx * dx + dz * dz)


def pick_radius_m() -> float:
    """Legacy floor-pick radius (kept for tests / callers)."""
    return 0.9


def _ray_sphere_t(
    origin: Vec3, direction: Vec3, center: Vec3, radius: float, *, max_t: float
) -> Optional[float]:
    oc = (origin[0] - center[0], origin[1] - center[1], origin[2] - center[2])
    # direction assumed unit
    b = oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2]
    c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    t0 = -b - s
    t1 = -b + s
    if t0 > 1e-5 and t0 <= max_t:
        return t0
    if t1 > 1e-5 and t1 <= max_t:
        return t1
    return None


def _ray_finite_cylinder_t(
    origin: Vec3,
    direction: Vec3,
    a0: Vec3,
    a1: Vec3,
    radius: float,
    up: str,
    *,
    max_t: float,
) -> Optional[float]:
    """Ray vs finite vertical cylinder between a0 and a1 (axis along up)."""
    # Project into horizontal plane for 2D circle test; constrain height along axis.
    if up == "Z":
        ox, oy = origin[0], origin[1]
        dx, dy = direction[0], direction[1]
        cx, cy = a0[0], a0[1]
        h0, h1 = a0[2], a1[2]
        oh, dh = origin[2], direction[2]
    else:
        ox, oy = origin[0], origin[2]
        dx, dy = direction[0], direction[2]
        cx, cy = a0[0], a0[2]
        h0, h1 = a0[1], a1[1]
        oh, dh = origin[1], direction[1]

    if h1 < h0:
        h0, h1 = h1, h0

    fx, fy = ox - cx, oy - cy
    a = dx * dx + dy * dy
    if a < 1e-12:
        # Ray parallel to axis — only spheres at ends can hit (handled separately)
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    inv = 0.5 / a
    for t in ((-b - s) * inv, (-b + s) * inv):
        if t <= 1e-5 or t > max_t:
            continue
        h = oh + t * dh
        if h0 - 1e-4 <= h <= h1 + 1e-4:
            return t
    return None


def _lift(floor_point: Vec3, amount: float, up: str) -> Vec3:
    x, y, z = floor_point
    if up == "Z":
        return (x, y, z + amount)
    return (x, y + amount, z)


def _as_vec2(value) -> Tuple[float, float]:
    if hasattr(value, "__len__") and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    if hasattr(value, "x") and hasattr(value, "y"):
        return (float(value.x), float(value.y))
    raise TypeError(f"Cannot convert to vec2: {type(value)}")


def _as_vec3(value) -> Optional[Vec3]:
    try:
        if hasattr(value, "__len__") and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
            return (float(value.x), float(value.y), float(value.z))
    except Exception:  # noqa: BLE001
        return None
    return None
