#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p logs/v51_tournament
exec 9>logs/v51_tournament/tournament.lock
flock -n 9 || { echo "Another V5.1 tournament is already running" >&2; exit 1; }
exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/run_v51_tournament.py "$@"
