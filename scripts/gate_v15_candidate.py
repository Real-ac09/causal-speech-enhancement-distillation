#!/usr/bin/env python3
"""Run the frozen V15 gate calculation with an explicit candidate label."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
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

    command = [
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
    ]
    subprocess.run(command, check=True)
    report_path = args.output_dir / "gate_report.json"
    report = json.loads(report_path.read_text())
    report["candidate"] = args.candidate_name
    report["candidate_label_wrapper"] = {
        "calculator": "scripts/gate_v15_quiet_level.py",
        "metrics_or_decisions_changed": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "candidate": args.candidate_name,
                "decision": report["decision"],
                "gate_summary": report["gate_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
