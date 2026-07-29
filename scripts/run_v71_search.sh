#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
SEARCH_BATCHES="${V71_SEARCH_BATCHES:-600}"
mkdir -p configs/v7/generated checkpoints/v71/search results/v71/search \
  results/v71/comparisons logs/v71
exec 9>logs/v71/search.lock
flock -n 9 || { echo "Another V7.1 search is active" >&2; exit 1; }

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
from pathlib import Path
import yaml

base = yaml.safe_load(Path("configs/v7/v71_teacher_search.yaml").read_text())
for name, use_full_band in (("multiscale_polar", False), ("fullband_polar", True)):
    config = yaml.safe_load(yaml.safe_dump(base))
    config["model"]["use_full_band"] = use_full_band
    config["paths"]["experiment_name"] = name
    Path(f"configs/v7/generated/v71_{name}.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
PY

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/check_v51_candidate.py \
  --config configs/v7/generated/v71_fullband_polar.yaml \
  --output results/v71/structural_gates.json --device cuda

for name in multiscale_polar fullband_polar; do
  config="configs/v7/generated/v71_$name.yaml"
  checkpoint="checkpoints/v71/search/$name/best.pt"
  output="results/v71/search/$name"
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

compare() {
  local reference="$1" candidate="$2" output="$3"
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "$reference" --candidate "$candidate" \
    --output-dir "$output" --bootstrap-samples 10000
}

compare v6_polar=results/v6/search/polar_phase_075 \
  v71_multiscale_polar=results/v71/search/multiscale_polar \
  results/v71/comparisons/v6_vs_multiscale_polar
compare v7_multiscale_complex=results/v7/search/multiscale \
  v71_multiscale_polar=results/v71/search/multiscale_polar \
  results/v71/comparisons/complex_vs_polar
compare v71_multiscale_polar=results/v71/search/multiscale_polar \
  v71_fullband_polar=results/v71/search/fullband_polar \
  results/v71/comparisons/multiscale_vs_fullband

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import json
from pathlib import Path

sources = {
    "v6_polar075": Path("results/v6/search/polar_phase_075"),
    "v7_multiscale_complex": Path("results/v7/search/multiscale"),
    "v71_multiscale_polar": Path("results/v71/search/multiscale_polar"),
    "v71_fullband_polar": Path("results/v71/search/fullband_polar"),
}
rows = []
for name, directory in sources.items():
    rows.append({
        "name": name,
        **json.loads((directory / "summary.json").read_text())["metrics"],
    })
winner = max(rows, key=lambda row: row["enhanced_pesq"])
report = {"winner": winner["name"], "candidates": rows}
Path("results/v71/search_decision.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
PY

echo "V7.1 controlled search complete"
