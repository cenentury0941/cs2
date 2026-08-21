# Overview

`crowd.agents` drives venue crowd simulation inside SimulCrowd.

## Modules

- `extension.py` — lifecycle and update loop
- `ui.py` — control panel
- `simulation.py` — per-frame orchestration
- `agent.py` — agent state
- `navigation.py` — NavMesh / path following
- `fire.py` — fire trigger and perception
- `ai.py` — decision engine interface

## Workflow

Open Venue → Generate NavMesh → Spawn Agents → Run → Trigger Fire → Observe
