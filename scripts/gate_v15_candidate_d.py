#!/usr/bin/env python3
"""Apply V15 gates with measured candidate-D deployment properties."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEPLOYMENT_NAMES = {
    "deployment_parameters",
    "deployment_algorithmic_latency_ms",
    "deployment_cpu_rtf_inherited_same_graph",
    "deployment_cpu_p95_ms_inherited_same_graph",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--cross-metadata", type=Path, required=True)
    parser.add_argument("--cross-reference", type=Path, required=True)
    parser.add_argument("--cross-candidate", type=Path, required=True)
    parser.add_argument("--voice-reference", type=Path, required=True)
    parser.add_argument("--voice-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "scripts/gate_v15_quiet_level.py",
            "--gates",
            str(args.gates),
            "--cross-metadata",
            str(args.cross_metadata),
            "--cross-reference",
            str(args.cross_reference),
            "--cross-candidate",
            str(args.cross_candidate),
            "--voice-reference",
            str(args.voice_reference),
            "--voice-candidate",
            str(args.voice_candidate),
            "--output-dir",
            str(args.output_dir),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
        ],
        check=True,
    )
    runtime = json.loads(args.runtime.read_text())
    gates = yaml.safe_load(args.gates.read_text())
    deployment = gates["deployment"]
    measured = {
        "deployment_parameters": (
            float(runtime["parameters"]),
            float(deployment["maximum_parameters"]),
        ),
        "deployment_algorithmic_latency_ms": (
            float(runtime["algorithmic_latency_ms"]),
            float(deployment["maximum_algorithmic_latency_ms"]),
        ),
        "deployment_cpu_rtf_inherited_same_graph": (
            float(runtime["streaming_rtf"]),
            float(deployment["maximum_cpu_rtf"]),
        ),
        "deployment_cpu_p95_ms_inherited_same_graph": (
            float(runtime["frame_time_ms"]["p95"]),
            float(deployment["maximum_cpu_p95_ms"]),
        ),
    }

    report_path = args.output_dir / "gate_report.json"
    report = json.loads(report_path.read_text())
    present = {decision["name"] for decision in report["gates"]}
    if not DEPLOYMENT_NAMES.issubset(present):
        raise ValueError("Base gate report is missing deployment decisions")
    for decision in report["gates"]:
        if decision["name"] not in measured:
            continue
        value, threshold = measured[decision["name"]]
        if decision["name"] == "deployment_parameters":
            value = int(value)
            threshold = int(threshold)
        decision.update(
            value=value,
            rule=f"<= {threshold}",
            passed=bool(value <= threshold),
        )
    failed = [
        decision for decision in report["gates"] if not decision["passed"]
    ]
    report["candidate"] = args.candidate_name
    report["deployment_measurement"] = {
        "runtime_report": str(args.runtime),
        "architecture": runtime["architecture"],
        "parameters": runtime["parameters"],
        "algorithmic_latency_ms": runtime["algorithmic_latency_ms"],
        "cpu_streaming_rtf": runtime["streaming_rtf"],
        "cpu_frame_time_p95_ms": runtime["frame_time_ms"]["p95"],
        "native_constant_work_streamer": True,
    }
    report["gate_summary"] = {
        "passed": len(report["gates"]) - len(failed),
        "failed": len(failed),
        "total": len(report["gates"]),
        "failed_names": [decision["name"] for decision in failed],
    }
    report["decision"] = (
        "promote_to_three_seed_replication"
        if not failed
        else "do_not_promote_candidate_budget_exhausted"
    )
    report["candidate_label_wrapper"] = {
        "calculator": "scripts/gate_v15_quiet_level.py",
        "development_metrics_changed": False,
        "deployment_values_replaced_with_candidate_measurements": True,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "candidate": args.candidate_name,
                "decision": report["decision"],
                "gate_summary": report["gate_summary"],
                "deployment": report["deployment_measurement"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
