#!/usr/bin/env bash
# Bootstrap and run the SimulCrowd SLM API on AWS Ubuntu (g6e / CUDA).
# Usage (on the instance):
#   cd /home/ubuntu/cs/services/slm
#   bash deploy_aws.sh
# Optional:
#   SLM_WORKERS=4 SLM_PORT=8080 SLM_MODEL_ID=microsoft/Phi-4-mini-instruct bash deploy_aws.sh
# Override SLM_WORKERS for N identical workers behind a round-robin LB on SLM_PORT.

set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

HOST="${SLM_HOST:-0.0.0.0}"
PORT="${SLM_PORT:-8080}"
WORKERS="${SLM_WORKERS:-4}"
MODEL="${SLM_MODEL_ID:-microsoft/Phi-4-mini-instruct}"
VENV="${SLM_VENV:-$ROOT/.venv-cuda}"

echo "==> Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script requires an NVIDIA GPU instance." >&2
  exit 1
fi
nvidia-smi -L

echo "==> Python venv: $VENV"
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  echo "python3-venv is missing. On Ubuntu run:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y python3.12-venv python3-pip" >&2
  exit 1
fi
if [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install --upgrade pip
# Drop stale remote modeling_* caches that break after transformers upgrades.
rm -rf "${HOME}/.cache/huggingface/modules/transformers_modules/microsoft" \
  "/home/ubuntu/.cache/huggingface/modules/transformers_modules/microsoft" 2>/dev/null || true
python -m pip install -r requirements-cuda.txt
# Ensure transformers is in the Phi-4-compatible window even if an older
# unconstrained install left 5.x in the venv.
python -m pip install 'transformers>=4.49.0,<4.54.0'

echo "==> Starting SLM fleet: workers=${WORKERS} public=${HOST}:${PORT} model=${MODEL}"
echo "    Workers bind 127.0.0.1:$((PORT + 1)).. ; LB round-robins on ${PORT}"
echo "    Local smoke:  curl -s http://127.0.0.1:${PORT}/health"
echo "    From laptop:  aws ssm start-session --target <instance-id> --region us-east-1 \\"
echo "      --document-name AWS-StartPortForwardingSession \\"
echo "      --parameters '{\"portNumber\":[\"${PORT}\"],\"localPortNumber\":[\"${PORT}\"]}'"
echo "    then:         curl -s http://127.0.0.1:${PORT}/ready"

exec python run_aws.py --host "$HOST" --port "$PORT" --workers "$WORKERS" --model "$MODEL"
