#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

CONFIG="configs/v5/generated/v51_recovery_main/adamw_loss_rescue_native_continuation.yaml"
SOURCE="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_loss_rescue_scratch/latest.pt"
RUN_DIR="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_loss_rescue_native_continuation"
SEED="$RUN_DIR/seed_epoch_018.pt"
OUTPUT="results/v51_optimizer_recovery/v51_recovery_main/native_continuation"
LOCKED400="data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
V43_REFERENCE="results/metrics/v5/apples_to_apples_locked400/v43_adaptive"

mkdir -p "$RUN_DIR" "$OUTPUT" logs/v51_optimizer_recovery
exec 9>logs/v51_optimizer_recovery/native_continuation.lock
flock -n 9 || { echo "Another native V5.1 continuation is active" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "Missing epoch-18 scratch checkpoint: $SOURCE" >&2; exit 1; }
if [[ ! -f "$SEED" ]]; then
    cp --reflink=auto "$SOURCE" "$SEED"
fi

if [[ -f "$RUN_DIR/latest.pt" ]]; then
    RESUME="$RUN_DIR/latest.pt"
else
    RESUME="$SEED"
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda --resume "$RESUME"

BEST="$RUN_DIR/best.pt"
[[ -f "$BEST" ]] || BEST="$RUN_DIR/latest.pt"
if [[ ! -f "$OUTPUT/best_native/summary.json" || ! -f "$OUTPUT/best_native/per_file_metrics.csv" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint "$BEST" --metadata "$LOCKED400" \
        --output-dir "$OUTPUT/best_native" --device cuda \
        --phase-residual-scale 1.0 --magnitude-residual-scale 1.0
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "v43=$V43_REFERENCE" \
    --candidate "native_continuation=$OUTPUT/best_native" \
    --output-dir "$OUTPUT/comparison" --bootstrap-samples 10000

echo "Native continuation complete: $OUTPUT/comparison/comparison.json"
