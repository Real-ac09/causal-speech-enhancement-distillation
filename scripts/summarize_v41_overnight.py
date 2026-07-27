#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("enhanced_pesq", "enhanced_si_sdr", "si_sdr_improvement", "enhanced_stoi", "enhanced_estoi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the V4.1 overnight comparison.")
    parser.add_argument("--muon-summary", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--ablation-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def metric_row(name: str, metrics: dict[str, Any], source: str) -> dict[str, Any]:
    # scripts/evaluate.py wraps aggregate values in a top-level ``metrics``
    # object, while the ablation evaluator already passes that object here.
    metrics = metrics.get("metrics", metrics)
    return {"name": name, "source": source, **{key: float(metrics[key]) for key in METRICS}}


def main() -> None:
    args = parse_args()
    muon = read_json(args.muon_summary)
    full = read_json(args.full_summary)
    control = read_json(args.control_summary)
    ablations = read_json(args.ablation_comparison)["ablations"]

    rows = [
        metric_row("muon_input", muon, "independently_trained"),
        metric_row("adaptive_vq_trained", full, "independently_trained"),
        metric_row("fixed2_equal_budget_trained", control, "independently_trained"),
    ]
    for name, result in ablations.items():
        rows.append(metric_row(f"checkpoint_{name}", result["metrics"], "inference_ablation"))
        rows[-1]["expected_iterations"] = float(result["expected_iterations"])
        rows[-1]["unique_codes"] = int(result["unique_codes"])
        rows[-1]["code_perplexity"] = float(result["global_code_perplexity"])

    reference = rows[0]
    for row in rows:
        row["delta_pesq_vs_muon_input"] = row["enhanced_pesq"] - reference["enhanced_pesq"]
        row["delta_si_sdr_vs_muon_input"] = row["enhanced_si_sdr"] - reference["enhanced_si_sdr"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "comparison.json").open("w") as file:
        json.dump({"reference": "muon_input", "results": rows}, file, indent=2)

    print("\nV4.1 overnight comparison")
    print(f"{'name':34s} {'PESQ':>8s} {'delta':>8s} {'SI-SDR':>9s} {'delta':>8s}")
    for row in rows:
        print(
            f"{row['name']:34s} {row['enhanced_pesq']:8.4f} "
            f"{row['delta_pesq_vs_muon_input']:+8.4f} {row['enhanced_si_sdr']:9.3f} "
            f"{row['delta_si_sdr_vs_muon_input']:+8.3f}"
        )
    print(f"\nWrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
