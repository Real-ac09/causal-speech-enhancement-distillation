#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

MODE="${1:-search}"
SEARCH_BATCHES="${V6_SEARCH_BATCHES:-600}"
mkdir -p logs/v6 results/v6 checkpoints/v6
exec 9>logs/v6/rework.lock
flock -n 9 || { echo "Another V6 programme is active" >&2; exit 1; }

run_smoke() {
    local config="configs/v6/v6_teacher_smoke.yaml"
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/check_v51_candidate.py --config "$config" \
        --output results/v6/structural_gates.json --device cuda
    if [[ ! -f checkpoints/v6/teacher_smoke/best.pt ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/train.py --config "$config" --device cuda \
            --max-train-batches 20 --max-val-batches 5
    fi
}

run_search() {
    local name="$1"
    local config="configs/v6/v6_teacher_search_${name}.yaml"
    local checkpoint_name
    case "$name" in
        complex) checkpoint_name="complex_ratio" ;;
        magnitude) checkpoint_name="magnitude_only" ;;
        polar) checkpoint_name="polar_residual" ;;
        *) echo "Unknown V6 search candidate: $name" >&2; return 2 ;;
    esac
    local checkpoint="checkpoints/v6/search/$checkpoint_name/best.pt"
    local output="results/v6/search/$checkpoint_name"
    if [[ ! -f "$checkpoint" ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/train.py --config "$config" --device cuda \
            --max-train-batches "$SEARCH_BATCHES"
    fi
    if [[ ! -f "$output/summary.json" ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/evaluate.py --checkpoint "$checkpoint" \
            --metadata data/processed/voicebank_demand/metadata/val_v51_search_100.csv \
            --output-dir "$output" --device cuda
    fi
}

select_winner() {
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import json
from pathlib import Path
import yaml

root = Path("results/v6/search")
names = ("complex_ratio", "magnitude_only", "polar_residual")
rows = []
for name in names:
    summary = json.loads((root / name / "summary.json").read_text())
    metrics = summary["metrics"]
    rows.append({"name": name, **metrics})
winner = max(rows, key=lambda row: row["enhanced_pesq"])
decision = {"winner": winner["name"], "candidates": rows}
Path("results/v6/search_winner.json").write_text(json.dumps(decision, indent=2) + "\n")

source_names = {
    "complex_ratio": "complex",
    "magnitude_only": "magnitude",
    "polar_residual": "polar",
}
source = Path(f"configs/v6/v6_teacher_search_{source_names[winner['name']]}.yaml")
config = yaml.safe_load(source.read_text())
config["project"]["seed"] = 6201
config["data"]["val_metadata"] = "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
config["training"]["epochs"] = 30
config["training"]["lr_scheduler"] = {
    "name": "reduce_on_plateau", "mode": "max",
    "metric": "perceptual_enhanced_pesq", "factor": 0.5,
    "patience": 3, "min_lr": 1e-5,
}
config["training"]["early_stopping"] = {
    "enabled": True, "patience": 6, "min_delta": 0.003,
}
config["training"]["perceptual_validation"]["max_items"] = 400
config["paths"]["checkpoint_dir"] = "checkpoints/v6"
config["paths"]["experiment_name"] = "teacher_full"
Path("configs/v6/generated").mkdir(parents=True, exist_ok=True)
Path("configs/v6/generated/v6_teacher_full.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False)
)
print(json.dumps(decision, indent=2))
print("Generated configs/v6/generated/v6_teacher_full.yaml")
PY
}

run_full() {
    local config="configs/v6/generated/v6_teacher_full.yaml"
    [[ -f "$config" ]] || { echo "Run the search stage before full training" >&2; exit 2; }
    if [[ ! -f checkpoints/v6/teacher_full/best.pt ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/train.py --config "$config" --device cuda
    fi
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint checkpoints/v6/teacher_full/best.pt \
        --metadata data/processed/voicebank_demand/metadata/val_v5_locked_400.csv \
        --output-dir results/v6/full_best_locked400 --device cuda \
        --phase-residual-scale 0.75 --magnitude-residual-scale 1.0
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/compare_evaluations.py \
        --reference v43=results/metrics/v5/apples_to_apples_locked400/v43_adaptive \
        --candidate v6=results/v6/full_best_locked400 \
        --output-dir results/v6/comparison --bootstrap-samples 10000
}

case "$MODE" in
    smoke)
        run_smoke
        ;;
    search)
        run_smoke
        run_search complex
        run_search magnitude
        run_search polar
        select_winner
        ;;
    full)
        run_full
        ;;
    all)
        run_smoke
        run_search complex
        run_search magnitude
        run_search polar
        select_winner
        run_full
        ;;
    *)
        echo "Usage: $0 {smoke|search|full|all}" >&2
        exit 2
        ;;
esac

echo "V6 $MODE stage complete"
