#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=src

CONFIG=configs/v10/train_v10_d40_full_seed8902.yaml
CHECKPOINT=checkpoints/v10/full/v10_d40_full_seed8902/best.pt
OUTPUT=results/v10/full/v10_d40_full_seed8902/locked400
COMPARISON=results/v10/full/v10_d40_full_seed8902/v8_comparison

mkdir -p checkpoints/v10/full results/v10/full logs/v10
exec 9>logs/v10/full.lock
flock -n 9 || { echo "Another V10 full run is active" >&2; exit 1; }

if [[ ! -f "$CHECKPOINT" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda
fi

if [[ ! -f "$OUTPUT/summary.json" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/evaluate.py --checkpoint "$CHECKPOINT" \
    --metadata data/processed/voicebank_demand/metadata/val_v5_locked_400.csv \
    --output-dir "$OUTPUT" --device cuda
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/compare_evaluations.py \
  --reference v8=results/v8/v8_direct_scalar_full_scratch/locked400_full \
  --candidate v10_d40="$OUTPUT" --output-dir "$COMPARISON" \
  --bootstrap-samples 20000
