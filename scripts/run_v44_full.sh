#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p checkpoints/v44/full results/v44/full logs/v44
exec 9>logs/v44/full.lock
flock -n 9 || { echo "Another V4.4 full run is active" >&2; exit 1; }

CONFIG=configs/v4/train_v44_causal_v43_recipe_full.yaml
CHECKPOINT=checkpoints/v44/full/causal_v43_recipe_full/best.pt
if [[ ! -f "$CHECKPOINT" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda
fi

OUTPUT=results/v44/full/locked400
if [[ ! -f "$OUTPUT/summary.json" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/evaluate.py --checkpoint "$CHECKPOINT" \
    --metadata data/processed/voicebank_demand/metadata/val_v5_locked_400.csv \
    --output-dir "$OUTPUT" --device cuda --phase-residual-scale 0.0
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/compare_evaluations.py \
  --reference v43_offline=results/metrics/v5/apples_to_apples_locked400/v43_adaptive \
  --candidate v44_causal="$OUTPUT" \
  --output-dir results/v44/full/v43_vs_v44 --bootstrap-samples 10000

echo "V4.4 full causal training and evaluation complete"
