#!/usr/bin/env python3
"""Apply the frozen V15 gates to the quiet-level candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METRICS = ("pesq", "si_sdr", "stoi", "estoi")


def _load(directory: Path) -> pd.DataFrame:
    return (
        pd.read_csv(directory / "per_file_metrics.csv")
        .set_index("file_id")
        .sort_index()
    )


def _bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--cross-metadata", type=Path, required=True)
    parser.add_argument("--cross-reference", type=Path, required=True)
    parser.add_argument("--cross-candidate", type=Path, required=True)
    parser.add_argument("--voice-reference", type=Path, required=True)
    parser.add_argument("--voice-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=15_014)
    args = parser.parse_args()

    gates = yaml.safe_load(args.gates.read_text())
    cross_metadata = (
        pd.read_csv(args.cross_metadata).set_index("file_id").sort_index()
    )
    cross_reference = _load(args.cross_reference)
    cross_candidate = _load(args.cross_candidate)
    voice_reference = _load(args.voice_reference)
    voice_candidate = _load(args.voice_candidate)
    if not (
        cross_metadata.index.equals(cross_reference.index)
        and cross_reference.index.equals(cross_candidate.index)
    ):
        raise ValueError("Cross-domain development IDs differ")
    if not voice_reference.index.equals(voice_candidate.index):
        raise ValueError("VoiceBank development IDs differ")

    rng = np.random.default_rng(args.bootstrap_seed)
    report: dict[str, object] = {
        "status": "complete",
        "gate_id": gates["gate_id"],
        "candidate": "v15_quiet_level_seed1200_epoch3",
        "reference": "v14_2_seed1200_epoch3",
        "external_test_used": False,
        "bootstrap": {
            "method": "paired_file",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "cross_domain": {"metrics": {}, "conditions": {}},
        "voicebank_development": {"metrics": {}},
        "gates": [],
    }
    cross_frame = cross_metadata.copy()

    for metric in METRICS:
        noisy_column = f"noisy_{metric}"
        enhanced_column = f"enhanced_{metric}"
        noisy_delta = (
            cross_candidate[noisy_column] - cross_reference[noisy_column]
        ).abs()
        if float(noisy_delta.max()) > 1e-7:
            raise ValueError(f"Cross-domain noisy {metric} values changed")
        candidate = cross_candidate[enhanced_column].to_numpy(float)
        reference = cross_reference[enhanced_column].to_numpy(float)
        noisy = cross_candidate[noisy_column].to_numpy(float)
        paired = candidate - reference
        gain = candidate - noisy
        cross_frame[f"candidate_{metric}_gain"] = gain
        cross_frame[f"candidate_minus_reference_{metric}"] = paired
        report["cross_domain"]["metrics"][metric] = {
            "noisy_mean": float(noisy.mean()),
            "reference_mean": float(reference.mean()),
            "candidate_mean": float(candidate.mean()),
            "candidate_gain_mean": float(gain.mean()),
            "candidate_gain_ci95": _bootstrap(
                gain, samples=args.bootstrap_samples, rng=rng
            ),
            "candidate_harm_rate": float((gain < 0.0).mean()),
            "candidate_minus_reference_mean": float(paired.mean()),
            "candidate_minus_reference_ci95": _bootstrap(
                paired, samples=args.bootstrap_samples, rng=rng
            ),
            "candidate_win_rate": float((paired > 0.0).mean()),
        }

        voice_noisy_delta = (
            voice_candidate[noisy_column] - voice_reference[noisy_column]
        ).abs()
        if float(voice_noisy_delta.max()) > 1e-7:
            raise ValueError(f"VoiceBank noisy {metric} values changed")
        voice_paired = (
            voice_candidate[enhanced_column]
            - voice_reference[enhanced_column]
        ).to_numpy(float)
        report["voicebank_development"]["metrics"][metric] = {
            "reference_mean": float(
                voice_reference[enhanced_column].mean()
            ),
            "candidate_mean": float(
                voice_candidate[enhanced_column].mean()
            ),
            "candidate_minus_reference_mean": float(voice_paired.mean()),
            "candidate_minus_reference_ci95": _bootstrap(
                voice_paired,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
            "candidate_win_rate": float((voice_paired > 0.0).mean()),
        }

    condition_rows: list[dict[str, object]] = []
    for condition in ("target_snr_db", "target_clean_rms_dbfs"):
        for value, group in cross_frame.groupby(condition, sort=True):
            row: dict[str, object] = {
                "condition": condition,
                "value": float(value),
                "items": int(len(group)),
            }
            for metric in METRICS:
                gain = group[f"candidate_{metric}_gain"]
                paired = group[f"candidate_minus_reference_{metric}"]
                row[f"candidate_{metric}_gain"] = float(gain.mean())
                row[f"candidate_{metric}_harm_rate"] = float(
                    (gain < 0.0).mean()
                )
                row[f"candidate_minus_reference_{metric}"] = float(
                    paired.mean()
                )
            condition_rows.append(row)
    conditions = pd.DataFrame(condition_rows)
    quietest = conditions[
        (conditions["condition"] == "target_clean_rms_dbfs")
        & (conditions["value"] == -35.0)
    ].iloc[0]
    report["cross_domain"]["conditions"]["quietest_clean_level"] = (
        quietest.to_dict()
    )

    decisions: list[dict[str, object]] = report["gates"]

    def minimum(name: str, value: float, threshold: float) -> None:
        decisions.append(
            {
                "name": name,
                "value": value,
                "rule": f">= {threshold}",
                "passed": bool(value >= threshold),
            }
        )

    def maximum(name: str, value: float, threshold: float) -> None:
        decisions.append(
            {
                "name": name,
                "value": value,
                "rule": f"<= {threshold}",
                "passed": bool(value <= threshold),
            }
        )

    cross_rules = gates["cross_domain_dev"]
    for metric, rule in cross_rules["candidate_minus_reference"].items():
        key = "si_sdr" if metric == "si_sdr_db" else metric
        result = report["cross_domain"]["metrics"][key]
        minimum(
            f"cross_candidate_minus_reference_{metric}_mean",
            result["candidate_minus_reference_mean"],
            float(rule["minimum_mean"]),
        )
        if "ci95_lower_must_exceed" in rule:
            threshold = float(rule["ci95_lower_must_exceed"])
            value = float(result["candidate_minus_reference_ci95"][0])
            decisions.append(
                {
                    "name": (
                        f"cross_candidate_minus_reference_{metric}_ci95_lower"
                    ),
                    "value": value,
                    "rule": f"> {threshold}",
                    "passed": bool(value > threshold),
                }
            )
    absolute = cross_rules["candidate_minus_noisy"]
    stoi = report["cross_domain"]["metrics"]["stoi"]
    minimum(
        "cross_candidate_minus_noisy_stoi_mean",
        stoi["candidate_gain_mean"],
        float(absolute["stoi"]["minimum_mean"]),
    )
    maximum(
        "cross_stoi_harm_rate",
        stoi["candidate_harm_rate"],
        float(absolute["stoi"]["maximum_harm_rate"]),
    )
    quiet_rule = absolute["quietest_clean_level_stoi"]
    minimum(
        "cross_quietest_stoi_gain_mean",
        float(quietest["candidate_stoi_gain"]),
        float(quiet_rule["minimum_mean"]),
    )
    maximum(
        "cross_quietest_stoi_harm_rate",
        float(quietest["candidate_stoi_harm_rate"]),
        float(quiet_rule["maximum_harm_rate"]),
    )

    voice_rules = gates["voicebank_development"][
        "candidate_minus_reference"
    ]
    for metric, rule in voice_rules.items():
        key = "si_sdr" if metric == "si_sdr_db" else metric
        minimum(
            f"voice_candidate_minus_reference_{metric}_mean",
            report["voicebank_development"]["metrics"][key][
                "candidate_minus_reference_mean"
            ],
            float(rule["minimum_mean"]),
        )

    deployment = gates["deployment"]
    maximum("deployment_parameters", 808095, deployment["maximum_parameters"])
    maximum(
        "deployment_algorithmic_latency_ms",
        20.0,
        deployment["maximum_algorithmic_latency_ms"],
    )
    maximum(
        "deployment_cpu_rtf_inherited_same_graph",
        0.372,
        deployment["maximum_cpu_rtf"],
    )
    maximum(
        "deployment_cpu_p95_ms_inherited_same_graph",
        4.077,
        deployment["maximum_cpu_p95_ms"],
    )
    failed = [decision for decision in decisions if not decision["passed"]]
    report["gate_summary"] = {
        "passed": len(decisions) - len(failed),
        "failed": len(failed),
        "total": len(decisions),
        "failed_names": [decision["name"] for decision in failed],
    }
    report["decision"] = (
        "promote_to_three_seed_replication"
        if not failed
        else "do_not_promote_continue_bounded_ablation"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cross_frame.reset_index().to_csv(
        args.output_dir / "per_file_analysis.csv", index=False
    )
    conditions.to_csv(args.output_dir / "condition_summary.csv", index=False)
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
