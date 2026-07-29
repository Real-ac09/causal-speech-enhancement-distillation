#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONFIG=configs/v4/generated/v45/promoted_scratch.yaml
CHECKPOINT=checkpoints/v45/full/promoted_scratch/best.pt
OUTPUT=results/v45/full/locked400

[[ -f "$CONFIG" ]] || { echo "No promoted V4.5 scratch config; run the recovery programme first" >&2; exit 2; }
mkdir -p checkpoints/v45/full results/v45/full logs/v45
exec 9>logs/v45/full.lock
flock -n 9 || { echo "Another V4.5 full run is active" >&2; exit 1; }

if [[ ! -f "$CHECKPOINT" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda
fi
if [[ ! -f "$OUTPUT/summary.json" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/evaluate.py --checkpoint "$CHECKPOINT" \
    --metadata data/processed/voicebank_demand/metadata/val_v5_locked_400.csv \
    --output-dir "$OUTPUT" --device cuda --phase-residual-scale 0.0
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/compare_evaluations.py \
  --reference v43_offline=results/metrics/v5/apples_to_apples_locked400/v43_adaptive \
  --candidate v45_causal="$OUTPUT" --output-dir results/v45/full/v43_vs_v45 \
  --bootstrap-samples 10000
