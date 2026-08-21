#!/usr/bin/env bash
# Bootstrap and run the SimulCrowd SLM API on AWS Ubuntu (g6e / CUDA).
# Usage (on the instance):
#   cd /home/ubuntu/cs/services/slm
#   bash deploy_aws.sh
# Optional: SLM_PORT=8080 SLM_MODEL_ID=microsoft/Phi-4-mini-instruct bash deploy_aws.sh

set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

HOST="${SLM_HOST:-0.0.0.0}"
PORT="${SLM_PORT:-8080}"
MODEL="${SLM_MODEL_ID:-microsoft/Phi-4-mini-instruct}"
VENV="${SLM_VENV:-$ROOT/.venv-cuda}"

echo "==> Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script requires an NVIDIA GPU instance." >&2
  exit 1
fi
nvidia-smi -L

echo "==> Python venv: $VENV"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt

echo "==> Starting SLM on ${HOST}:${PORT} (model=${MODEL})"
echo "    Local smoke:  curl -s http://127.0.0.1:${PORT}/health"
echo "    From laptop:  aws ssm start-session --target <instance-id> --region us-east-1 \\"
echo "      --document-name AWS-StartPortForwardingSession \\"
echo "      --parameters '{\"portNumber\":[\"${PORT}\"],\"localPortNumber\":[\"${PORT}\"]}'"
echo "    then:         curl -s http://127.0.0.1:${PORT}/ready"

exec python run_aws.py --host "$HOST" --port "$PORT" --model "$MODEL"
