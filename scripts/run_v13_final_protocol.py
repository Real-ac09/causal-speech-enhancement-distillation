#!/usr/bin/env python3
"""Verify or execute the frozen V13 final evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("configs/v13/frozen_final_protocol.yaml")
RESULT_ROOT = Path("results/v13/final_standard_test")
EXPECTED_MODEL_FIELDS = {
    "architecture": "causal_temporal_core_v12",
    "variant": "student",
    "sample_rate": 16000,
    "n_fft": 512,
    "win_length": 320,
    "hop_length": 160,
    "auxiliary_vq": False,
    "use_mamba": False,
    "temporal_core": "gru",
    "temporal_hidden_dim": 232,
    "time_kernel_size": 1,
    "phase_residual_scale": 0.0,
    "reconstruction_mode": "direct_scalar_mask",
    "scale_preserving_detail": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(relative_path: str, expected: str) -> None:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Frozen input changed: {relative_path}\n"
            f"expected {expected}\n"
            f"actual   {actual}"
        )


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        return sum(1 for _ in reader)


def _load_and_verify_protocol() -> tuple[dict[str, Any], str]:
    protocol_file = ROOT / PROTOCOL_PATH
    protocol = yaml.safe_load(protocol_file.read_text())
    if protocol.get("status") != "frozen":
        raise ValueError("The final protocol must have status: frozen")

    for evidence in protocol["selection_evidence"].values():
        _require_digest(evidence["path"], evidence["sha256"])
    for source, digest in protocol["source_files"].items():
        _require_digest(source, digest)

    seen_seeds: set[int] = set()
    for checkpoint in protocol["checkpoints"]:
        seed = int(checkpoint["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate checkpoint seed: {seed}")
        seen_seeds.add(seed)
        _require_digest(checkpoint["path"], checkpoint["sha256"])
        _require_digest(checkpoint["config"], checkpoint["config_sha256"])
        config = yaml.safe_load((ROOT / checkpoint["config"]).read_text())
        if int(config["project"]["seed"]) != seed:
            raise ValueError(f"Config seed mismatch for seed {seed}")
        for field, expected in EXPECTED_MODEL_FIELDS.items():
            actual = config["model"].get(field)
            if actual != expected:
                raise ValueError(
                    f"Frozen model mismatch in {checkpoint['config']}: "
                    f"{field} expected {expected!r}, got {actual!r}"
                )
    if seen_seeds != {1200, 1201, 1202}:
        raise ValueError(f"Expected seeds 1200-1202, got {sorted(seen_seeds)}")

    standard_test = protocol["standard_test"]
    _require_digest(
        standard_test["metadata"], standard_test["metadata_sha256"]
    )
    rows = _csv_rows(ROOT / standard_test["metadata"])
    if rows != int(standard_test["items"]):
        raise ValueError(
            f"Standard-test row count changed: expected "
            f"{standard_test['items']}, got {rows}"
        )
    return protocol, _sha256(protocol_file)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    python = Path(sys.executable)
    libraries = [str(python.parent.parent / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    environment["PYTHONPATH"] = "src"
    return environment


def _run(command: list[str], execute: bool, complete: bool) -> None:
    if complete:
        print("= existing output:", " ".join(command), flush=True)
        return
    print("+", " ".join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def _validate_completed_outputs(
    protocol: dict[str, Any],
    directories: list[Path],
    aggregate: Path,
    runtime_output: Path,
) -> None:
    expected_items = int(protocol["standard_test"]["items"])
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if int(summary["num_items"]) != expected_items:
            raise ValueError(
                f"{directory} contains {summary['num_items']} items; "
                f"expected {expected_items}"
            )
        if summary["weights"] != "model" or summary["chunk_seconds"] is not None:
            raise ValueError(
                f"{directory} is not a full-utterance, raw-model evaluation"
            )
        if int(summary["hop_length"]) != int(protocol["model"]["hop_length"]):
            raise ValueError(f"{directory} used an unexpected hop length")

    aggregate_report = json.loads(aggregate.read_text())
    if int(aggregate_report["num_seeds"]) != len(directories):
        raise ValueError("Final aggregate has an unexpected seed count")
    if int(aggregate_report["num_files_per_seed"]) != expected_items:
        raise ValueError("Final aggregate has an unexpected file count")

    runtime = json.loads(runtime_output.read_text())
    gate = protocol["evaluation"]["runtime_gate"]
    if gate["p95_below_hop_deadline"] and not runtime[
        "passes_hop_deadline_p95"
    ]:
        raise RuntimeError("Final model failed the frozen p95 runtime gate")
    if gate["p99_below_hop_deadline"] and not runtime[
        "passes_hop_deadline_p99"
    ]:
        raise RuntimeError("Final model failed the frozen p99 runtime gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Evaluation device; GPU use must be requested explicitly.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, verify the freeze and print commands only.",
    )
    parser.add_argument(
        "--acknowledge-standard-test",
        action="store_true",
        help=(
            "Confirm that the standard test will be used only for the frozen "
            "protocol and will not drive architecture or checkpoint reselection."
        ),
    )
    args = parser.parse_args()
    if args.execute and not args.acknowledge_standard_test:
        parser.error("--execute requires --acknowledge-standard-test")

    protocol, protocol_digest = _load_and_verify_protocol()
    standard_test = protocol["standard_test"]
    evaluation = protocol["evaluation"]
    directories: list[Path] = []
    for checkpoint in protocol["checkpoints"]:
        seed = int(checkpoint["seed"])
        output = ROOT / RESULT_ROOT / f"seed{seed}"
        directories.append(output)
        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            checkpoint["path"],
            "--metadata",
            standard_test["metadata"],
            "--output-dir",
            str(output.relative_to(ROOT)),
            "--device",
            args.device,
            "--weights",
            "model",
        ]
        _run(command, args.execute, (output / "summary.json").is_file())

    aggregate = ROOT / RESULT_ROOT / "aggregate_three_seed.json"
    aggregate_command = [
        sys.executable,
        "scripts/aggregate_final_model_seeds.py",
    ]
    for directory in directories:
        aggregate_command.extend(
            ["--evaluation", str(directory.relative_to(ROOT))]
        )
    aggregate_command.extend(
        [
            "--model-name",
            protocol["research_label"],
            "--expected-files",
            str(standard_test["items"]),
            "--bootstrap-samples",
            str(evaluation["bootstrap_samples"]),
            "--bootstrap-seed",
            str(evaluation["bootstrap_seed"]),
            "--output",
            str(aggregate.relative_to(ROOT)),
        ]
    )
    _run(aggregate_command, args.execute, aggregate.is_file())

    runtime_checkpoint = next(
        checkpoint
        for checkpoint in protocol["checkpoints"]
        if int(checkpoint["seed"]) == int(evaluation["runtime_checkpoint_seed"])
    )
    runtime_output = ROOT / "results/runtime/v13_final_gru_seed1200_cpu.json"
    runtime_command = [
        sys.executable,
        "scripts/measure_latency.py",
        "--checkpoint",
        runtime_checkpoint["path"],
        "--warmup-seconds",
        str(evaluation["runtime_warmup_seconds"]),
        "--seconds",
        str(evaluation["runtime_measurement_seconds"]),
        "--seed",
        str(evaluation["runtime_checkpoint_seed"]),
        "--output",
        str(runtime_output.relative_to(ROOT)),
    ]
    _run(runtime_command, args.execute, runtime_output.is_file())

    if args.execute:
        expected_outputs = [
            *(directory / "summary.json" for directory in directories),
            aggregate,
            runtime_output,
        ]
        missing = [str(path) for path in expected_outputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"Final protocol outputs are missing: {missing}")
        _validate_completed_outputs(
            protocol,
            directories,
            aggregate,
            runtime_output,
        )
        execution_record = {
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": protocol_digest,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "device_for_quality_evaluation": args.device,
            "architecture_reselection_permitted": False,
            "checkpoint_reselection_permitted": False,
            "outputs": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in expected_outputs
            },
        }
        record_path = ROOT / RESULT_ROOT / "execution_record.json"
        record_path.write_text(json.dumps(execution_record, indent=2) + "\n")
        print(json.dumps(execution_record, indent=2))


if __name__ == "__main__":
    main()
