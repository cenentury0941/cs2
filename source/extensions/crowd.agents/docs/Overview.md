# Overview

`crowd.agents` drives venue crowd simulation inside SimulCrowd.

## Modules

- `extension.py` — lifecycle and update loop
- `ui.py` — Crowd Sim panel + Individual Agent Info
- `simulation.py` — per-frame orchestration
- `inspect.py` — viewport click/hover → inspect agent (no lasting USD selection)
- `viz.py` — highlight colors + selected-agent FOV wireframe
- `agent.py` — agent state
- `navigation.py` — NavMesh / path following
- `fire.py` — fire trigger and perception
- `ai.py` — decision engine interface

## Workflow

Open Venue → Generate NavMesh → Spawn Agents → click agent (info panel) → Run → Trigger Fire → Observe
