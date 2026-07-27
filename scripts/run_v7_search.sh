#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
SEARCH_BATCHES="${V7_SEARCH_BATCHES:-600}"
mkdir -p configs/v7/generated checkpoints/v7/search results/v7/search logs/v7
exec 9>logs/v7/search.lock
flock -n 9 || { echo "Another V7 search is active" >&2; exit 1; }

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
from pathlib import Path
import yaml

base = yaml.safe_load(Path("configs/v7/v7_teacher_search.yaml").read_text())
variants = {
    "multiscale": {"use_full_band": False, "adaptive_phase": False},
    "fullband": {"use_full_band": True, "adaptive_phase": False},
    "adaptive": {"use_full_band": False, "adaptive_phase": True},
    "combined": {"use_full_band": True, "adaptive_phase": True},
}
root = Path("configs/v7/generated")
root.mkdir(parents=True, exist_ok=True)
for name, settings in variants.items():
    config = yaml.safe_load(yaml.safe_dump(base))
    config["model"].update(settings)
    config["paths"]["experiment_name"] = name
    (root / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/check_v51_candidate.py \
    --config configs/v7/generated/combined.yaml \
    --output results/v7/structural_gates.json --device cuda

for name in multiscale fullband adaptive combined; do
    config="configs/v7/generated/$name.yaml"
    checkpoint="checkpoints/v7/search/$name/best.pt"
    output="results/v7/search/$name"
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
done

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import json
from pathlib import Path

names = ("multiscale", "fullband", "adaptive", "combined")
rows = []
for name in names:
    metrics = json.loads(
        (Path("results/v7/search") / name / "summary.json").read_text()
    )["metrics"]
    rows.append({"name": name, **metrics})
baseline = rows[0]
for row in rows:
    row["delta_pesq"] = row["enhanced_pesq"] - baseline["enhanced_pesq"]
    row["delta_si_sdr"] = row["enhanced_si_sdr"] - baseline["enhanced_si_sdr"]
    row["delta_stoi"] = row["enhanced_stoi"] - baseline["enhanced_stoi"]
    row["delta_estoi"] = row["enhanced_estoi"] - baseline["enhanced_estoi"]
    row["passes_no_harm"] = (
        row["delta_si_sdr"] >= -0.15
        and row["delta_stoi"] >= -0.002
        and row["delta_estoi"] >= -0.003
    )
eligible = [row for row in rows if row["passes_no_harm"]]
winner = max(eligible, key=lambda row: row["enhanced_pesq"])
report = {"baseline": "multiscale", "winner": winner["name"], "candidates": rows}
Path("results/v7/search_decision.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

echo "V7 controlled search complete"
