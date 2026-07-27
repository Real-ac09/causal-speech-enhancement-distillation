#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-conda}"
FORCE=0
DRY_RUN=0
for argument in "$@"; do
    case "$argument" in
        --force) FORCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done

ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/v41_overnight_ablation"
DONE_DIR="$LOG_DIR/done"
RESULT_ROOT="results/metrics/v41_overnight_ablation"
CONTROL_CHECKPOINT="checkpoints/noise_adaptive_tf_mamba_v41_fixed2_muon_equal_budget/best.pt"
CONTROL_LATEST="checkpoints/noise_adaptive_tf_mamba_v41_fixed2_muon_equal_budget/latest.pt"
MUON_CHECKPOINT="checkpoints/noise_adaptive_tf_mamba_v41_muon_finetune/best.pt"
FULL_CHECKPOINT="checkpoints/noise_adaptive_tf_mamba_v41_segment_vq_muon/best.pt"
TEST_METADATA="data/processed/voicebank_demand/metadata/test.csv"

mkdir -p "$LOG_DIR" "$DONE_DIR" "$RESULT_ROOT"
exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.1 overnight pipeline is already running." >&2
    exit 1
fi

run_step() {
    local name="$1"
    shift
    echo
    echo "[$(date --iso-8601=seconds)] START $name"
    if (( DRY_RUN )); then
        printf '  %q' "$@"
        printf '\n'
        return
    fi
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    echo "[$(date --iso-8601=seconds)] DONE  $name"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file is missing: $1" >&2
        exit 1
    fi
}

require_file "$MUON_CHECKPOINT"
require_file "$FULL_CHECKPOINT"
require_file "$TEST_METADATA"

TRAIN_DONE="$DONE_DIR/train_fixed2_equal_budget.done"
if (( FORCE )) || [[ ! -f "$TRAIN_DONE" ]]; then
    resume_arguments=()
    if (( ! FORCE )) && [[ -f "$CONTROL_LATEST" ]]; then
        resume_arguments=(--resume "$CONTROL_LATEST")
        echo "RESUME training from $CONTROL_LATEST"
    fi
    run_step train_fixed2_equal_budget \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/train.py \
        --config configs/v4/train_v41_fixed2_muon_equal_budget.yaml \
        --device cuda \
        "${resume_arguments[@]}"
    if (( ! DRY_RUN )); then
        touch "$TRAIN_DONE"
    fi
else
    echo "SKIP training: completion marker $TRAIN_DONE exists"
fi

if (( ! DRY_RUN )); then
    require_file "$CONTROL_CHECKPOINT"
fi

MUON_SUMMARY="results/metrics/noise_adaptive_tf_mamba_v41_muon_finetune_test_full/summary.json"
if (( FORCE )) || [[ ! -f "$MUON_SUMMARY" ]]; then
    run_step evaluate_muon_input \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint "$MUON_CHECKPOINT" \
        --metadata "$TEST_METADATA" \
        --output-dir results/metrics/noise_adaptive_tf_mamba_v41_muon_finetune_test_full \
        --device cuda
else
    echo "SKIP evaluation: $MUON_SUMMARY already exists"
fi

FULL_SUMMARY="results/metrics/noise_adaptive_tf_mamba_v41_segment_vq_muon_test_full/summary.json"
if (( FORCE )) || [[ ! -f "$FULL_SUMMARY" ]]; then
    run_step evaluate_adaptive_vq_trained \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint "$FULL_CHECKPOINT" \
        --metadata "$TEST_METADATA" \
        --output-dir results/metrics/noise_adaptive_tf_mamba_v41_segment_vq_muon_test_full \
        --device cuda
else
    echo "SKIP evaluation: $FULL_SUMMARY already exists"
fi

CONTROL_SUMMARY="$RESULT_ROOT/fixed2_equal_budget/summary.json"
if (( FORCE )) || [[ ! -f "$CONTROL_SUMMARY" ]]; then
    run_step evaluate_fixed2_equal_budget \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint "$CONTROL_CHECKPOINT" \
        --metadata "$TEST_METADATA" \
        --output-dir "$RESULT_ROOT/fixed2_equal_budget" \
        --device cuda
else
    echo "SKIP evaluation: $CONTROL_SUMMARY already exists"
fi

ABLATION_COMPARISON="$RESULT_ROOT/checkpoint_ablations/comparison.json"
if (( FORCE )) || [[ ! -f "$ABLATION_COMPARISON" ]]; then
    run_step evaluate_checkpoint_ablations \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate_v4_ablations.py --checkpoint "$FULL_CHECKPOINT" \
        --metadata "$TEST_METADATA" \
        --output-dir "$RESULT_ROOT/checkpoint_ablations" \
        --device cuda \
        --modes full no_vq no_condition fixed_1 fixed_2 fixed_4
else
    echo "SKIP ablations: $ABLATION_COMPARISON already exists"
fi

if (( DRY_RUN )); then
    echo "Dry run complete; no commands were executed."
    exit 0
fi

require_file "$MUON_SUMMARY"
require_file "$FULL_SUMMARY"
require_file "$CONTROL_SUMMARY"
require_file "$ABLATION_COMPARISON"
run_step summarize \
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/summarize_v41_overnight.py \
    --muon-summary "$MUON_SUMMARY" \
    --full-summary "$FULL_SUMMARY" \
    --control-summary "$CONTROL_SUMMARY" \
    --ablation-comparison "$ABLATION_COMPARISON" \
    --output-dir "$RESULT_ROOT"

echo
echo "[$(date --iso-8601=seconds)] OVERNIGHT PIPELINE COMPLETE"
echo "Summary: $RESULT_ROOT/comparison.csv"
