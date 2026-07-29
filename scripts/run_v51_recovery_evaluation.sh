#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

METADATA="data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
OUTPUT_ROOT="results/v51_optimizer_recovery/v51_recovery_main/full_utterance"
SOURCE_CHECKPOINT="checkpoints/v51_tournament/v51_main/arch_v51_log_256/best.pt"
MUON_CHECKPOINT="checkpoints/v51_optimizer_recovery/v51_recovery_main/muon_3e4_loss_rescue/best.pt"
ADAMW_CHECKPOINT="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_1e4_loss_rescue/best.pt"

mkdir -p "$OUTPUT_ROOT" logs/v51_optimizer_recovery
exec 9>logs/v51_optimizer_recovery/evaluation.lock
flock -n 9 || { echo "Another V5.1 recovery evaluation is active" >&2; exit 1; }

evaluate_one() {
    local checkpoint="$1"
    local output="$2"
    if [[ -f "$output/summary.json" && -f "$output/per_file_metrics.csv" ]]; then
        echo "Skipping completed evaluation: $output"
        return
    fi
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$checkpoint" \
        --metadata "$METADATA" \
        --output-dir "$output" \
        --device cuda
}

evaluate_one "$SOURCE_CHECKPOINT" "$OUTPUT_ROOT/source"
evaluate_one "$MUON_CHECKPOINT" "$OUTPUT_ROOT/muon_loss_rescue"
evaluate_one "$ADAMW_CHECKPOINT" "$OUTPUT_ROOT/adamw_loss_rescue"

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "source=$OUTPUT_ROOT/source" \
    --candidate "muon_loss_rescue=$OUTPUT_ROOT/muon_loss_rescue" \
    --candidate "adamw_loss_rescue=$OUTPUT_ROOT/adamw_loss_rescue" \
    --output-dir "$OUTPUT_ROOT/comparison" \
    --bootstrap-samples 10000

echo "Full-utterance recovery evaluation complete: $OUTPUT_ROOT/comparison/comparison.json"
