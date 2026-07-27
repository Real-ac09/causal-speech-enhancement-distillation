#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=src

mkdir -p logs/v11
exec 9>logs/v11/programme.lock
flock -n 9 || { echo "Another V11 programme is active" >&2; exit 1; }

if pgrep -f '[s]cripts/train.py' >/dev/null; then
  echo "An existing training job is active" >&2
  exit 2
fi

exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/run_v11_magnitude_recovery.py "$@"
