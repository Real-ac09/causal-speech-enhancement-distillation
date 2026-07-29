#!/usr/bin/env python3
"""Train and evaluate the gated 16-file V8.6 BF16/FP32 comparison."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/v8/generated/v86_promoted"
CHECKPOINT_ROOT = ROOT / "checkpoints/v86"
RESULT_ROOT = ROOT / "results/v86/precision_comparison"
METADATA = ROOT / "data/processed/voicebank_demand/metadata/v82_overfit_16.csv"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)
CANDIDATES = ("ratio_16_bf16", "ratio_16_fp32")


def run(*command: str) -> None:
    environment = os.environ.copy()
    libraries = [str(ENV_PREFIX / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def python(*arguments: str) -> None:
    run(str(CONDA), "run", "--no-capture-output", "-n", ENV_NAME, "python", *arguments)


def evaluate(name: str) -> dict[str, object]:
    output = RESULT_ROOT / name
    summary_path = output / "summary.json"
    if not summary_path.exists():
        python(
            "scripts/evaluate_v5_reconstruction_ablations.py",
            "--checkpoint", str((CHECKPOINT_ROOT / name / "best.pt").relative_to(ROOT)),
            "--metadata", str(METADATA.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--max-items", "16", "--chunk-seconds", "4.0", "--device", "cuda",
        )
    summary = json.loads(summary_path.read_text())
    fitted = summary["estimated_magnitude_noisy_phase"]
    return {
        "name": name,
        "precision": name.rsplit("_", 1)[-1],
        "pesq": fitted["enhanced_pesq"],
        "stoi": fitted["enhanced_stoi"],
        "estoi": fitted["enhanced_estoi"],
        "si_sdr": fitted["enhanced_si_sdr"],
    }


def write_status(results: list[dict[str, object]], complete: bool) -> None:
    ranking = sorted(results, key=lambda row: float(row["pesq"]), reverse=True)
    status = {
        "complete": complete,
        "completed_candidates": len(results),
        "total_candidates": len(CANDIDATES),
        "v82_reference_pesq": 2.573778599500656,
        "capacity_gate_pesq": 2.8,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
        "capacity_gate": bool(ranking and float(ranking[0]["pesq"]) >= 2.8),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


def main() -> None:
    results = []
    for name in CANDIDATES:
        config = CONFIG_ROOT / f"{name}.yaml"
        checkpoint = CHECKPOINT_ROOT / name / "best.pt"
        if not checkpoint.exists():
            python("scripts/train.py", "--config", str(config.relative_to(ROOT)), "--device", "cuda")
        results.append(evaluate(name))
        write_status(results, complete=False)
    write_status(results, complete=True)


if __name__ == "__main__":
    main()
