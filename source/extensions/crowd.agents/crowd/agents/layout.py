# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Startup workspace layout for SimulCrowd."""

from __future__ import annotations

import asyncio
from typing import Optional

import carb
import omni.kit.app
import omni.ui as ui

from .ui import AGENT_INFO_TITLE, WINDOW_TITLE

# Built-in Kit window titles
_VIEWPORT = "Viewport"
_STAGE = "Stage"
_PROPERTY = "Property"
_RENDER_SETTINGS = "Render Settings"
_TOOLBAR = "Main ToolBar"
_CONSOLE = "Console"
_CONTENT = "Content"

_SHOW_TITLES = (_VIEWPORT, _STAGE, _PROPERTY, _RENDER_SETTINGS, _TOOLBAR, WINDOW_TITLE, AGENT_INFO_TITLE)
_HIDE_TITLES = (_CONSOLE, _CONTENT)


async def apply_startup_layout(retries: int = 40) -> None:
    """Dock windows into: Agent Info | Viewport | (tabs / Crowd Sim).

    Layout (left → right):
      1. Individual Agent Info
      2. 3D Viewport (+ toolbar)
      3. Right column:
           - tabs: Stage, Render Settings, Property
           - Crowd Sim (below)

    Console and Content are hidden on launch (still available from Window menu).
    """
    app = omni.kit.app.get_app()
    settings = carb.settings.get_settings()
    if settings.get("/app/testMode") or settings.get("/exts/crowd.agents/layout/enabled") is False:
        return

    for _ in range(10):
        await app.next_update_async()

    for title in _SHOW_TITLES:
        try:
            ui.Workspace.show_window(title)
        except Exception:  # noqa: BLE001
            pass

    _hide_default_panels()

    columns_ok = False
    for attempt in range(retries):
        await app.next_update_async()
        if _dock_columns():
            columns_ok = True
            carb.log_info(f"[crowd.agents] Column layout applied (attempt {attempt + 1})")
            break

    if not columns_ok:
        carb.log_warn("[crowd.agents] Startup layout incomplete — column windows missing")
        return

    # Property defaults to deferred_dock_in("Details") which does not exist in
    # kit_base_editor, so it floats. Render Settings targets Stage, but that can
    # race our Stage move. Re-tab both onto Stage after columns settle.
    for attempt in range(retries):
        await app.next_update_async()
        if _tab_onto_stage():
            # Give deferred_dock_in a few frames to resolve
            for _ in range(6):
                await app.next_update_async()
            if _tabs_share_stage_dock():
                carb.log_info(f"[crowd.agents] Stage tabs docked (attempt {attempt + 1})")
                break
        # Keep retrying — windows may appear late
    else:
        carb.log_warn(
            "[crowd.agents] Could not tab Property/Render Settings onto Stage — "
            "check Window menu titles"
        )

    # Split Crowd under the tab group only after tabs are attached
    for attempt in range(10):
        await app.next_update_async()
        if _dock_crowd_under_stage():
            carb.log_info(f"[crowd.agents] Crowd Sim docked under Stage (attempt {attempt + 1})")
            break

    # Final pass: re-assert tabs in case the Crowd split disturbed them
    await app.next_update_async()
    _tab_onto_stage()
    for _ in range(4):
        await app.next_update_async()

    # Content/Console often re-show after docking; hide again at the end.
    _hide_default_panels()


def _get(title: str) -> Optional[ui.Window]:
    try:
        return ui.Workspace.get_window(title)
    except Exception:  # noqa: BLE001
        return None


def _hide_default_panels() -> None:
    """Hide Console and Content so they don't take layout space."""
    for title in _HIDE_TITLES:
        try:
            ui.Workspace.show_window(title, False)
        except Exception:  # noqa: BLE001
            pass
        win = _get(title)
        if win is not None:
            try:
                win.visible = False
            except Exception:  # noqa: BLE001
                pass


def _dock_columns() -> bool:
    viewport = _get(_VIEWPORT)
    agent_info = _get(AGENT_INFO_TITLE)
    stage = _get(_STAGE)
    crowd = _get(WINDOW_TITLE)
    toolbar = _get(_TOOLBAR)

    if not viewport or not agent_info or not stage or not crowd:
        return False

    if toolbar:
        toolbar.visible = True
        toolbar.dock_in_window(_VIEWPORT, ui.DockPosition.LEFT, 0.03)

    agent_info.visible = True
    agent_info.dock_in_window(_VIEWPORT, ui.DockPosition.LEFT, 0.18)

    stage.visible = True
    stage.dock_in_window(_VIEWPORT, ui.DockPosition.RIGHT, 0.28)
    return True


def _tab_onto_stage() -> bool:
    """Tab Property + Render Settings onto Stage via deferred_dock_in."""
    stage = _get(_STAGE)
    if not stage:
        return False
    stage.visible = True

    ok = False
    for title, policy in (
        (_RENDER_SETTINGS, ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE),
        (_PROPERTY, ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE),
    ):
        win = _get(title)
        if not win:
            continue
        win.visible = True
        # Prefer handle-based SAME dock when available; fall back to deferred.
        docked = False
        try:
            if hasattr(win, "dock_in") and stage is not None:
                docked = bool(win.dock_in(stage, ui.DockPosition.SAME))
        except Exception:  # noqa: BLE001
            docked = False
        if not docked:
            try:
                win.deferred_dock_in(_STAGE, policy)
                docked = True
            except Exception as exc:  # noqa: BLE001
                carb.log_warn(f"[crowd.agents] deferred_dock_in failed for {title}: {exc}")
        ok = ok or docked

    try:
        stage.selected_in_dock = True
    except Exception:  # noqa: BLE001
        pass
    return ok


def _tabs_share_stage_dock() -> bool:
    stage = _get(_STAGE)
    if not stage:
        return False
    try:
        stage_id = stage.dock_id
    except Exception:  # noqa: BLE001
        return False
    if not stage_id:
        return False

    matched = 0
    expected = 0
    for title in (_PROPERTY, _RENDER_SETTINGS):
        win = _get(title)
        if win is None:
            continue
        expected += 1
        try:
            if win.docked and win.dock_id == stage_id:
                matched += 1
        except Exception:  # noqa: BLE001
            pass
    # Success if every present window shares Stage's dock, or at least one did
    # when both are present.
    if expected == 0:
        return False
    return matched == expected


def _dock_crowd_under_stage() -> bool:
    stage = _get(_STAGE)
    crowd = _get(WINDOW_TITLE)
    if not stage or not crowd:
        return False
    crowd.visible = True
    crowd.dock_in_window(_STAGE, ui.DockPosition.BOTTOM, 0.48)
    return True


def schedule_startup_layout() -> asyncio.Task:
    return asyncio.ensure_future(apply_startup_layout())
