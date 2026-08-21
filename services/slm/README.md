# SimulCrowd SLM service (Mac / mlx_lm)

Standalone HTTP inference for SimulCrowd. Runs outside Omniverse on Apple Silicon
via [mlx-lm](https://github.com/ml-explore/mlx-lm). Omniverse posts batches here;
this process never imports Kit or USD.

## Contract

| Endpoint | Behavior |
|---|---|
| `GET /health` | Process up |
| `GET /ready` | `200` only after the model is loaded |
| `POST /v1/decide` | Batched agent decisions |

Allowed intents: `GO_TO_SEAT`, `EXIT`, `IDLE`.

See repo root `agent.md` for the frozen request/response JSON.

## Mac setup (one instance)

```bash
cd services/slm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

First start downloads `mlx-community/Phi-4-mini-instruct-4bit` (~2 GB) into the
Hugging Face cache, then listens on `http://127.0.0.1:8080`.

Environment overrides:

- `SLM_MODEL_ID` — default `mlx-community/Phi-4-mini-instruct-4bit`
- `SLM_HOST` / `SLM_PORT` — bind address (default `127.0.0.1:8080`)
- `SLM_TEMPERATURE` / `SLM_MAX_TOKENS` — sampling (default `0.2` / `100`)

## Smoke test

With the server running in another terminal:

```bash
cd services/slm
source .venv/bin/activate
python smoke_test.py
```

## AWS / Ubuntu later

Mac uses `mlx_lm` as the worker backend. Cloud deploy will keep the same
`/health`, `/ready`, and `/v1/decide` contract and swap the worker for a CUDA
stack (e.g. vLLM) plus multiple instances and load distribution. Do not change
the HTTP schema unilaterally.
