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
| `POST /v1/summarize` | Narrative summary of one agent's Agent Info snapshot + events (**OpenRouter** when configured; else local model) |

Allowed intents: `GO_TO_SEAT`, `EXIT`, `IDLE`.

### `/v1/summarize` (OpenRouter)

Decide stays on the local Phi worker. Summaries use OpenRouter when a key is available:

- Env: `OPENROUTER_API_KEY`
- Or file: repo-root `openrouter_api_key.txt` (`API Key: sk-or-v1-…`)
- Model default: `openai.gpt-5.6-luna` → API id `openai/gpt-5.6-luna` (`OPENROUTER_MODEL`)

Request mirrors the Individual Agent Info panel:

```json
{
  "request_id": "sum-1",
  "agent_id": "Agent_001",
  "snapshot": {
    "id": "Agent_001",
    "intent": "EXIT",
    "goal": "Exit_A",
    "assigned_seat": "Seat_012",
    "exit_chosen": "Exit_A",
    "fire_perceived": "yes",
    "smoke_perceived": "no",
    "at_destination": "no",
    "position": "(1.00, 0.00, 2.00)",
    "destination": "(10.00, 0.00, 0.00)",
    "waypoints_left": "3"
  },
  "events": [
    "t=12.3s  Agent_001: saw fire",
    "t=10.0s  Agent_001: intent — GO_TO_SEAT → EXIT"
  ]
}
```

Response:

```json
{
  "request_id": "sum-1",
  "agent_id": "Agent_001",
  "summary": "Agent_001 is evacuating toward Exit_A after perceiving fire…"
}
```

See repo root `agent.md` for the frozen decide request/response JSON.

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
- `SLM_TEMPERATURE` / `SLM_MAX_TOKENS` — sampling for `/v1/decide` (default `0.2` / `100`)
- `SLM_SUMMARIZE_TEMPERATURE` / `SLM_SUMMARIZE_MAX_TOKENS` — sampling for `/v1/summarize` (default `0.4` / `256`)
- `OPENROUTER_API_KEY` — required for hosted summaries (else local model)
- `OPENROUTER_MODEL` — default `openai.gpt-5.6-luna`
- `OPENROUTER_BASE_URL` / `OPENROUTER_TIMEOUT_S` — default OpenRouter API + 60s

## Smoke test

With the server running in another terminal:

```bash
cd services/slm
source .venv/bin/activate
python smoke_test.py
```

## AWS Ubuntu (CUDA)

Same HTTP contract; worker is Hugging Face `transformers` on NVIDIA GPU
(not mlx). On the g6e.xlarge instance:

```bash
cd /home/ubuntu/cs/services/slm
bash deploy_aws.sh
```

Or manually:

```bash
cd services/slm
python3 -m venv .venv-cuda
source .venv-cuda/bin/activate
pip install -r requirements-cuda.txt
# If pip's default torch is CPU-only, install a CUDA wheel first, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
python run_aws.py
```

Defaults: bind `0.0.0.0:8080`, model `microsoft/Phi-4-mini-instruct`,
`SLM_BACKEND=cuda` (one process; Phi-4-mini bf16 uses ~10 GB VRAM idle).

`run_aws.py` / `deploy_aws.sh` start a single CUDA uvicorn on the public port.
Clients use `http://host:8080`. `/ready` is 200 after the model has loaded.

Reach the API from a laptop without opening the security group:

```bash
aws ssm start-session \
  --target <instance-id> --region us-east-1 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'
```

Then `curl http://127.0.0.1:8080/health` / `python smoke_test.py`.

## Behavior / scenario tests

Rule + parse unit tests (no server):

```bash
cd services/slm
python -m unittest tests.test_decision_parse tests.test_summarize tests.test_openrouter -v
```

Live model behavior across agent states (server must be up):

```bash
cd services/slm
python tests/test_behavior_live.py --base http://127.0.0.1:8080
```

Scenarios live in `tests/scenarios.py` (fire, smoke, seated, lost, exit goal, etc.).

