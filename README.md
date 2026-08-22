# SimulCrowd

NVIDIA Omniverse / Kit application for intent-driven venue crowd simulation.
Agents walk a NavMesh to seats, perceive fire and smoke, and evacuate through
exits. High-level decisions can come from a local rule engine or a hosted
small language model (SLM) over HTTP — Omniverse never loads model weights.

Built on the Omniverse Kit App Template / Kit Base Editor stack.

## Features

**Venue & navigation**
- Import venue geometry (GLB → USD) and bake a walkable NavMesh
- Scene contract: spawn plane, numbered seats, exits, and Nav include volumes
- Spawn 1–100 agents with unique seat assignment and path planning on Run
- Corridor lane jitter, waypoint skip/replan, and simple agent–agent separation

**Perception & evacuation**
- Place a fire trigger in the viewport (hover preview, click-to-place)
- Per-agent FOV / range / mesh line-of-sight for fire and smoke (sticky once seen)
- Intent switch to EXIT only for agents that actually perceive the hazard
- Debug colors: blue toward seat, red toward exit

**Operator tools**
- Crowd Sim control panel (open venue, NavMesh, spawn, run/pause/reset, fire, SLM)
- Click/hover inspect for Individual Agent Info (intent, goal, seat, exit,
  fire/smoke, position, waypoints)
- Selected-agent FOV wireframe and state-change event log for the run

**AI / decisioning**
- Pluggable decision engines: rule-based (default) or remote SLM
- Batched HTTP intents with fallback to rules if the SLM is down
- Optional natural-language agent summaries (OpenRouter or local model)

## Architecture

SimulCrowd splits **simulation** (inside Kit) from **inference** (outside Kit):

```text
┌─────────────────────────────────────────────────────────┐
│  Omniverse Kit — SimulCrowd (crowd.sim.kit)             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  crowd.agents                                     │  │
│  │  · Perception (FOV / LOS / sticky memory)         │  │
│  │  · DecisionEngine (rules  or  RemoteSLM client)   │  │
│  │  · Navigation (NavMesh paths, separation, USD)    │  │
│  │  · Fire + UI / agent inspect                      │  │
│  └───────────────────────┬───────────────────────────┘  │
└──────────────────────────┼──────────────────────────────┘
                           │  HTTP  /v1/decide  (batch)
                           │  HTTP  /v1/summarize
                           ▼
┌─────────────────────────────────────────────────────────┐
│  services/slm  (standalone process)                     │
│  Mac: mlx-lm (Phi-4-mini)  ·  AWS: CUDA transformers    │
│  Optional: OpenRouter for /v1/summarize                 │
└─────────────────────────────────────────────────────────┘
```

**Per-frame loop (Kit):** perception updates locally every frame → navigation
follows the current intent → decisions are gated (not every frame when using
SLM). Between SLM ticks, agents keep the last intent while NavMesh motion
continues.

**Labour split:** Kit owns scene, perception scalars, pathing, and USD sync.
The SLM receives compact per-agent JSON (ids, bools, goals — no meshes) and
returns only high-level intents. On timeout or `/ready` failure, Kit falls
back to `RuleBasedDecisionEngine`.

## AI features

| Capability | How it works |
|------------|--------------|
| **Rule-based decisions** | Local engine in `crowd.agents`; default when SLM is off or unreachable |
| **Hosted SLM decisions** | `POST /v1/decide` batches agents; model chooses `GO_TO_SEAT`, `EXIT`, or `IDLE` from perception + goals |
| **Perception-driven prompts** | Decides fire on perception *changes* (and batch size), not a blind all-agents poll every frame |
| **Safe batching** | One bad agent parse never fails the whole batch; soft/hard motion throttle under SLM backlog |
| **Agent summaries** | `POST /v1/summarize` turns an Agent Info snapshot + events into operator-facing narrative (OpenRouter when configured, else local Phi) |
| **Deploy options** | Mac Apple Silicon via mlx-lm, or AWS Ubuntu CUDA (`Phi-4-mini`); LAN or SSM port-forward to Windows Kit |

Allowed intents are frozen: **`GO_TO_SEAT`**, **`EXIT`**, **`IDLE`**. See
`services/slm/README.md` and `context/Omniverse_SLM_Technical_Flow.txt` for
the HTTP contract and end-to-end flow.

## Repository layout

| Path | Purpose |
|------|---------|
| `source/apps/crowd.sim.kit` | SimulCrowd Kit application |
| `source/extensions/crowd.agents/` | Crowd sim extension (UI, NavMesh, fire, SLM client) |
| `services/slm/` | Standalone HTTP SLM inference (Mac mlx / AWS CUDA) |
| `context/` | Setup notes (LAN SLM, Omniverse↔SLM flow, progress) |
| `templates/` | Upstream Kit app/extension templates (scaffolding) |
| `tools/` | Repo tooling |

## Prerequisites

- **OS:** Windows 10/11 or Linux (Ubuntu 22.04+)
- **GPU:** NVIDIA RTX (3070 or better recommended)
- **Driver:** ≥550.54.15 (Linux) or ≥551.78 (Windows)
- **Git**, and the usual Kit SDK toolchain (see
  [Windows developer configuration](readme-assets/additional-docs/windows_developer_configuration.md)
  if building C++ on Windows)

Apple Silicon is used only for the optional Mac SLM service (`services/slm`),
not for running the Omniverse app.

## Quick start

### Build

**Linux:**
```bash
./repo.sh build
```

**Windows:**
```powershell
.\repo.bat build
```

### Launch

**Linux:**
```bash
./repo.sh launch
```

**Windows:**
```powershell
.\repo.bat launch
```

Select **SimulCrowd** / `crowd.sim`. First startup can take several minutes
while shaders compile.

### Typical workflow

In the Crowd Sim panel:

1. **Open Venue** → **Generate NavMesh** → **Spawn Agents**
2. Click an agent to inspect state
3. **Run** → agents walk to seats
4. **Start Fire** (click to place) → agents that see fire/smoke exit
5. Optionally enable **Use hosted SLM**, set the base URL, **Check Ready**, then Run

Scene naming for venues:

- `Spawn` — spawn plane
- `Seat_001` … `Seat_100` — seat planes
- `Exit_A`, `Exit_B` — exits
- `Nav_001` / `Nav.001` … — converted to NavMesh include volumes

## Hosted SLM (optional)

Decisions stay in Kit by default (rule-based). For SLM-backed intents, run the
standalone service and point the Crowd Sim panel at it:

```bash
cd services/slm
# see services/slm/README.md for Mac (mlx) or AWS (CUDA) setup
```

Contract: `GET /health`, `GET /ready`, `POST /v1/decide`, `POST /v1/summarize`.
LAN wiring notes: `context/SLM_LAN_Connection.txt`.

## Extension docs

- [crowd.agents overview](source/extensions/crowd.agents/docs/Overview.md)
- [crowd.agents README](source/extensions/crowd.agents/docs/README.md)

## Governing terms

The software and materials are governed by the
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the
[Product-Specific Terms for NVIDIA Omniverse](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-omniverse/).

The Omniverse Kit SDK may collect anonymous usage data; see
[data collection details](readme-assets/additional-docs/data_collection_and_use.md).

## Additional resources

- [Usage and Troubleshooting](readme-assets/additional-docs/usage_and_troubleshooting.md)
- [Kit SDK Tooling Guide](readme-assets/additional-docs/kit_app_template_tooling_guide.md)
- [Omniverse Kit SDK Manual](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/index.html)
