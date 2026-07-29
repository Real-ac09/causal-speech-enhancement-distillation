#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
RUN=("$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python)
LOCKED="data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
LOG_DIR="logs/v5_programme"
mkdir -p "$LOG_DIR" results/metrics/v5
exec 9>"$LOG_DIR/pipeline.lock"
flock -n 9 || { echo "Another V5 programme is running" >&2; exit 1; }

step() {
    local name="$1"; shift
    echo "[$(date --iso-8601=seconds)] START $name"
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
    echo "[$(date --iso-8601=seconds)] DONE $name"
}
train() {
    local name="$1" config="$2" directory="$3"
    local resume=()
    [[ -f "$directory/latest.pt" ]] && resume=(--resume "$directory/latest.pt")
    step "$name" "${RUN[@]}" scripts/train.py --config "$config" --device cuda "${resume[@]}"
}
evaluate() {
    local name="$1" checkpoint="$2" metadata="$3" output="$4"
    step "$name" "${RUN[@]}" scripts/evaluate.py --checkpoint "$checkpoint" \
        --metadata "$metadata" --output-dir "$output" --device cuda
}

step validation_subset "${RUN[@]}" scripts/create_v5_validation_subset.py \
    --input data/processed/voicebank_demand/metadata/val.csv --output "$LOCKED"
train teacher_smoke configs/v5/train_v5_teacher_smoke.yaml checkpoints/v5_teacher_smoke
train teacher_foundation configs/v5/train_v5_teacher_foundation.yaml checkpoints/v5_teacher_foundation
evaluate foundation_locked checkpoints/v5_teacher_foundation/best.pt "$LOCKED" results/metrics/v5/foundation_locked
evaluate foundation_full checkpoints/v5_teacher_foundation/best.pt \
    data/processed/voicebank_demand/metadata/val.csv results/metrics/v5/foundation_full
step save_quality_examples "${RUN[@]}" scripts/save_v5_enhanced_examples.py \
    --checkpoint checkpoints/v5_teacher_foundation/best.pt \
    --metadata data/processed/voicebank_demand/metadata/train.csv \
    --output-dir results/quality_training/v5_foundation --max-items 1000
step quality_regressor "${RUN[@]}" scripts/train_v5_quality_regressor.py \
    --metadata data/processed/voicebank_demand/metadata/train.csv \
    --enhanced-dir results/quality_training/v5_foundation \
    --max-files 3000 --output checkpoints/v5_quality_regressor/best.pt
train teacher_perceptual configs/v5/train_v5_teacher_perceptual.yaml checkpoints/v5_teacher_perceptual
evaluate perceptual_locked checkpoints/v5_teacher_perceptual/best.pt "$LOCKED" results/metrics/v5/perceptual_locked
evaluate perceptual_full checkpoints/v5_teacher_perceptual/best.pt \
    data/processed/voicebank_demand/metadata/val.csv results/metrics/v5/perceptual_full
step listening_examples "${RUN[@]}" scripts/save_v5_enhanced_examples.py \
    --checkpoint checkpoints/v5_teacher_perceptual/best.pt --metadata "$LOCKED" \
    --output-dir results/listening/v5_fixed_20 --max-items 20
step squim_safeguard "${RUN[@]}" scripts/evaluate_v5_squim.py \
    --checkpoint checkpoints/v5_teacher_perceptual/best.pt --metadata "$LOCKED" \
    --output results/metrics/v5/perceptual_squim.json --max-items 400
train bounded_adapter configs/v5/train_v5_teacher_bounded_adapter.yaml checkpoints/v5_teacher_bounded_adapter
evaluate adapter_locked checkpoints/v5_teacher_bounded_adapter/best.pt "$LOCKED" results/metrics/v5/adapter_locked
step compare_adapter "${RUN[@]}" scripts/compare_evaluations.py \
    --reference train_only=results/metrics/v5/perceptual_locked \
    --candidate bounded_adapter=results/metrics/v5/adapter_locked \
    --output-dir results/metrics/v5/adapter_gate
step gate_adapter "${RUN[@]}" scripts/gate_v5_vq_adapter.py \
    --comparison results/metrics/v5/adapter_gate/comparison.json \
    --output results/metrics/v5/adapter_gate/decision.json
step distill_student "${RUN[@]}" scripts/train_distill.py \
    --config configs/v5/distill_v5_student.yaml
evaluate student_locked checkpoints/v5_student_distilled/best.pt "$LOCKED" results/metrics/v5/student_locked
step runtime_teacher "${RUN[@]}" scripts/benchmark_v5_runtime.py \
    --checkpoint checkpoints/v5_teacher_perceptual/best.pt --output results/metrics/v5/teacher_runtime.json --seconds 0.25
step runtime_student "${RUN[@]}" scripts/benchmark_v5_runtime.py \
    --checkpoint checkpoints/v5_student_distilled/best.pt --output results/metrics/v5/student_runtime.json --seconds 0.25

if [[ "${V5_RUN_PCS:-0}" == "1" ]]; then
    train teacher_pcs configs/v5/train_v5_teacher_pcs.yaml checkpoints/v5_teacher_pcs_separate
    evaluate pcs_full checkpoints/v5_teacher_pcs_separate/best.pt \
        data/processed/voicebank_demand/metadata/val.csv results/metrics/v5/pcs_full
fi

echo "V5 architecture-selection programme complete. Test-set evaluation remains deliberately locked."
