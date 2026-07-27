#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="src"

WAIT_FOR_IDLE=false
PREPARE_ONLY=false
FORWARD_ARGS=()
for argument in "$@"; do
  if [[ "$argument" == "--wait-for-idle" ]]; then
    WAIT_FOR_IDLE=true
  else
    FORWARD_ARGS+=("$argument")
    if [[ "$argument" == "--prepare-only" ]]; then
      PREPARE_ONLY=true
    fi
  fi
done

mkdir -p logs/v10
exec 9>logs/v10/programme.lock
flock -n 9 || { echo "Another V10 programme is active" >&2; exit 1; }

if [[ "$PREPARE_ONLY" == true ]]; then
  :
elif [[ "$WAIT_FOR_IDLE" == true ]]; then
  while pgrep -f '[s]cripts/train.py' >/dev/null; do
    printf '%s Waiting for active training jobs to finish...\n' "$(date --iso-8601=seconds)"
    sleep 60
  done
elif pgrep -f '[s]cripts/train.py' >/dev/null; then
  echo "An existing training job is active; rerun with --wait-for-idle or after it finishes." >&2
  exit 2
fi

exec "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/run_v10_programme.py "${FORWARD_ARGS[@]}"
