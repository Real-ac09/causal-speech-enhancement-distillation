#!/usr/bin/env python3
"""Verify or execute the frozen V14.2 final evaluation protocol."""

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

import torch
import yaml

from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("configs/v14/frozen_final_protocol.yaml")
RESULT_ROOT = Path("results/v14/final_standard_test")
RUNTIME_OUTPUT = Path("results/runtime/v14_2_distilled_seed1200_cpu.json")
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

    checkpoint = protocol["checkpoint"]
    _require_digest(checkpoint["path"], checkpoint["sha256"])
    _require_digest(
        checkpoint["training_config"], checkpoint["training_config_sha256"]
    )
    payload = torch.load(
        ROOT / checkpoint["path"], map_location="cpu", weights_only=False
    )
    if int(payload["epoch"]) != int(checkpoint["epoch"]):
        raise ValueError("Frozen checkpoint epoch does not match the protocol")
    model_config = payload["config"]["model"]
    for field, expected in EXPECTED_MODEL_FIELDS.items():
        actual = model_config.get(field)
        if actual != expected:
            raise ValueError(
                f"Frozen model mismatch: {field} expected {expected!r}, "
                f"got {actual!r}"
            )
    model = build_model(model_config)
    model.load_state_dict(payload["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(protocol["model"]["parameter_count"]):
        raise ValueError(
            f"Parameter count changed: expected "
            f"{protocol['model']['parameter_count']}, got {parameter_count}"
        )

    reference = protocol["reference"]
    _require_digest(reference["summary"], reference["summary_sha256"])
    _require_digest(
        reference["per_file_metrics"], reference["per_file_metrics_sha256"]
    )

    standard_test = protocol["standard_test"]
    _require_digest(
        standard_test["metadata"], standard_test["metadata_sha256"]
    )
    expected_items = int(standard_test["items"])
    if _csv_rows(ROOT / standard_test["metadata"]) != expected_items:
        raise ValueError("Standard-test row count changed")
    if _csv_rows(ROOT / reference["per_file_metrics"]) != expected_items:
        raise ValueError("Frozen V13 reference row count changed")
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


def _validate_outputs(
    protocol: dict[str, Any],
    quality_directory: Path,
    comparison_directory: Path,
    runtime_output: Path,
) -> None:
    expected_items = int(protocol["standard_test"]["items"])
    summary = json.loads((quality_directory / "summary.json").read_text())
    if int(summary["num_items"]) != expected_items:
        raise ValueError("Final quality evaluation has an unexpected item count")
    if summary["weights"] != "model" or summary["chunk_seconds"] is not None:
        raise ValueError("Final evaluation is not raw-model, full-utterance")
    if int(summary["hop_length"]) != int(protocol["model"]["hop_length"]):
        raise ValueError("Final evaluation used an unexpected hop length")

    comparison = json.loads(
        (comparison_directory / "comparison.json").read_text()
    )
    if "v14_2" not in comparison["paired_bootstrap"]:
        raise ValueError("Paired V13 comparison is missing")

    runtime = json.loads(runtime_output.read_text())
    if int(runtime["parameters"]) != int(protocol["model"]["parameter_count"]):
        raise ValueError("Runtime benchmark parameter count changed")
    gate = protocol["evaluation"]["runtime_gate"]
    if gate["p95_below_hop_deadline"] and not runtime[
        "passes_hop_deadline_p95"
    ]:
        raise RuntimeError("V14.2 failed the frozen p95 runtime gate")
    if gate["p99_below_hop_deadline"] and not runtime[
        "passes_hop_deadline_p99"
    ]:
        raise RuntimeError("V14.2 failed the frozen p99 runtime gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--acknowledge-standard-test",
        action="store_true",
        help=(
            "Confirm that standard-test results will not drive architecture, "
            "loss, recipe, or checkpoint reselection."
        ),
    )
    args = parser.parse_args()
    if args.execute and not args.acknowledge_standard_test:
        parser.error("--execute requires --acknowledge-standard-test")

    protocol, protocol_digest = _load_and_verify_protocol()
    checkpoint = protocol["checkpoint"]
    standard_test = protocol["standard_test"]
    evaluation = protocol["evaluation"]
    reference = protocol["reference"]

    quality_directory = ROOT / RESULT_ROOT / "seed1200"
    quality_command = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint",
        checkpoint["path"],
        "--metadata",
        standard_test["metadata"],
        "--output-dir",
        str(quality_directory.relative_to(ROOT)),
        "--device",
        args.device,
        "--weights",
        "model",
    ]
    _run(
        quality_command,
        args.execute,
        (quality_directory / "summary.json").is_file(),
    )

    comparison_directory = ROOT / RESULT_ROOT / "v14_2_vs_v13_seed1200"
    comparison_command = [
        sys.executable,
        "scripts/compare_evaluations.py",
        "--reference",
        f"v13={Path(reference['summary']).parent}",
        "--candidate",
        f"v14_2={quality_directory.relative_to(ROOT)}",
        "--output-dir",
        str(comparison_directory.relative_to(ROOT)),
        "--bootstrap-samples",
        str(evaluation["bootstrap_samples"]),
        "--seed",
        str(evaluation["bootstrap_seed"]),
    ]
    _run(
        comparison_command,
        args.execute,
        (comparison_directory / "comparison.json").is_file(),
    )

    runtime_output = ROOT / RUNTIME_OUTPUT
    runtime_command = [
        sys.executable,
        "scripts/measure_latency.py",
        "--checkpoint",
        checkpoint["path"],
        "--warmup-seconds",
        str(evaluation["runtime_warmup_seconds"]),
        "--seconds",
        str(evaluation["runtime_measurement_seconds"]),
        "--seed",
        str(checkpoint["seed"]),
        "--output",
        str(runtime_output.relative_to(ROOT)),
    ]
    _run(runtime_command, args.execute, runtime_output.is_file())

    if not args.execute:
        return
    expected_outputs = [
        quality_directory / "summary.json",
        quality_directory / "per_file_metrics.csv",
        comparison_directory / "comparison.json",
        comparison_directory / "comparison.csv",
        runtime_output,
    ]
    missing = [str(path) for path in expected_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Final protocol outputs are missing: {missing}")
    _validate_outputs(
        protocol, quality_directory, comparison_directory, runtime_output
    )
    execution_record = {
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_digest,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_for_quality_evaluation": args.device,
        "single_seed_upgrade": True,
        "architecture_reselection_permitted": False,
        "loss_or_recipe_reselection_permitted": False,
        "checkpoint_reselection_permitted": False,
        "standard_test_items": int(standard_test["items"]),
        "outputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in expected_outputs
        },
    }
    record_path = ROOT / RESULT_ROOT / "execution_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(execution_record, indent=2) + "\n")
    print(json.dumps(execution_record, indent=2))


if __name__ == "__main__":
    main()
