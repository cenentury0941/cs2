# Agent brief — SimulCrowd

Copy this file for the other agent and change **only** the Role section.

## Role

**You are in charge of hosted SLM instances.**

Own the standalone inference service that runs outside Omniverse (Mac M2 locally, cloud GPU later). Do not work on the Kit app, USD, NavMesh, agents, fire VFX, or the in-app simulation loop.

## Division of labour

Two agents work in parallel. Stay on your side of the boundary.

| Hosted SLM instances | Omniverse / Kit |
|---|---|
| Standalone HTTP inference service | `crowd.agents` extension |
| Batch `POST /v1/decide` API | `RemoteSLMDecisionEngine` HTTP client |
| 1–N workers of the **same** small model | Perception JSON from agent state |
| Prompt + constrained JSON output | Map `intent` → NavMesh |
| Health / ready / load-balance / fallback | Simulation loop, seats, fire, VFX |
| Runs on Mac M2 or cloud GPU | Runs on RTX laptop; **does not** load the model |

The laptop (4 GB VRAM) must not host the SLM. Omniverse never imports the model — it only posts a batch and applies returned intents.

### Shared contract (do not change unilaterally)

Keep `DecisionEngine.decide(agent) -> intent`. Intents are **only**:

- `GO_TO_SEAT`
- `EXIT`
- `IDLE`

Navigation owns movement. The SLM does not emit waypoints, mesh, or per-frame motion.

Rule-based behaviour to match (and later exceed with variety): fire visible → `EXIT`; at destination → `IDLE`; otherwise seat/goal → `GO_TO_SEAT`.

### Frozen HTTP API

`POST /v1/decide` — one request may contain all agents (batch). Not one call per agent.

Request:

```json
{
  "request_id": "sim-tick-42",
  "agents": [
    {
      "id": "Agent_001",
      "goal": "Seat_003",
      "assigned_seat": "Seat_003",
      "intent": "WALK",
      "perception": {
        "fire_visible": false,
        "smoke_visible": false,
        "seat_known": true,
        "at_destination": false,
        "nearest_exit": "Exit_A",
        "exits": ["Exit_A", "Exit_B"]
      }
    }
  ]
}
```

Response:

```json
{
  "request_id": "sim-tick-42",
  "decisions": [
    {
      "id": "Agent_001",
      "intent": "GO_TO_SEAT",
      "destination": "Seat_003"
    }
  ]
}
```

Also: `GET /health`, `GET /ready` (200 only after a worker has the model). Invalid model output for one agent must not 500 the batch.

Tiny JSON only. No USD, no NavMesh, no raw geometry.

### Layout

- SLM service: `services/slm/` (outside Kit; runnable on Mac without Omniverse)
- Omniverse: `source/apps/crowd.sim.kit`, `source/extensions/crowd.agents/`

### Do not

- One SLM process per agent — batch on a worker
- Reason every frame — scheduler on the Omniverse side (~1 Hz)
- Touch the other agent's files unless the shared contract must change (agree first)
