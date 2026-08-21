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
`SLM_BACKEND=cuda`, **`SLM_WORKERS=4`** (~32–40 GB VRAM for Phi-4-mini bf16).

`run_aws.py` / `deploy_aws.sh` start N identical worker processes on loopback
(`8081`, `8082`, …) and a round-robin load balancer on the public port. Clients
keep using `http://host:8080`. Override when needed:

```bash
SLM_WORKERS=1 bash deploy_aws.sh   # lighter / share GPU with Omniverse
SLM_WORKERS=4 bash deploy_aws.sh   # default
# or: python run_aws.py --workers 4
```

Each worker loads its own model copy into GPU memory. On a shared Omniverse/DCV
box, watch `nvidia-smi` — 4 workers leave little headroom on a 48 GB L40S.

`/ready` is 200 when **at least one** worker has finished loading; the JSON may
also include `workers_ready` / `workers_total`.

Reach the API from a laptop without opening the security group:

```bash
aws ssm start-session \
  --target <instance-id> --region us-east-1 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'
```

Then `curl http://127.0.0.1:8080/health` / `python smoke_test.py`.

Do not change the `/v1/decide` schema unilaterally.
