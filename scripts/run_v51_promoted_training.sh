#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

CONFIG="configs/v5/generated/v51_recovery_main/adamw_loss_rescue_promoted.yaml"
SOURCE="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_1e4_loss_rescue_guarded_1000/latest.pt"
RUN_DIR="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_loss_rescue_promoted"
SEED="$RUN_DIR/seed_epoch_002.pt"
OUTPUT="results/v51_optimizer_recovery/v51_recovery_main/promoted"
LOCKED400="data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
FULL_VAL="data/processed/voicebank_demand/metadata/val.csv"

mkdir -p "$RUN_DIR" "$OUTPUT" logs/v51_optimizer_recovery
exec 9>logs/v51_optimizer_recovery/promoted.lock
flock -n 9 || { echo "Another promoted V5.1 run is active" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "Missing guarded checkpoint: $SOURCE" >&2; exit 1; }
if [[ ! -f "$SEED" ]]; then
    cp --reflink=auto "$SOURCE" "$SEED"
fi

evaluate() {
    local checkpoint="$1" metadata="$2" output="$3"
    if [[ ! -f "$output/summary.json" || ! -f "$output/per_file_metrics.csv" ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/evaluate.py --checkpoint "$checkpoint" --metadata "$metadata" \
            --output-dir "$output" --device cuda \
            --phase-residual-scale 0.0 --magnitude-residual-scale 0.95
    fi
}

# Establish comparable baselines before consuming a long training budget.
evaluate "$SEED" "$LOCKED400" "$OUTPUT/baseline_locked400"
evaluate "$SEED" "$FULL_VAL" "$OUTPUT/baseline_full_val"

if [[ -f "$RUN_DIR/latest.pt" ]]; then
    RESUME="$RUN_DIR/latest.pt"
else
    RESUME="$SEED"
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda --resume "$RESUME"

BEST="$RUN_DIR/best.pt"
[[ -f "$BEST" ]] || BEST="$RUN_DIR/latest.pt"
evaluate "$BEST" "$LOCKED400" "$OUTPUT/best_locked400"

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "seed=$OUTPUT/baseline_locked400" \
    --candidate "promoted_best=$OUTPUT/best_locked400" \
    --output-dir "$OUTPUT/comparison" --bootstrap-samples 10000

echo "Promoted training and evaluation complete: $OUTPUT/comparison/comparison.json"
