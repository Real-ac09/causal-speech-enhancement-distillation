#!/usr/bin/env python3
"""Run the frozen V14.2 seed-1201/1202 replication protocol."""

from __future__ import annotations

import argparse
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
PROTOCOL_PATH = Path("configs/v14/frozen_replication_protocol.yaml")
RECORD_PATH = Path("results/v14/replication/training_execution_record.json")
EXPECTED_MODEL_FIELDS = {
    "architecture": "causal_temporal_core_v12",
    "variant": "student",
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


def _require_digest(path_text: str, expected: str) -> None:
    path = ROOT / path_text
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Frozen input changed: {path_text}\n"
            f"expected {expected}\n"
            f"actual   {actual}"
        )


def _load_protocol() -> tuple[dict[str, Any], str]:
    protocol_path = ROOT / PROTOCOL_PATH
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise ValueError("Replication protocol must have status: frozen")
    selection = protocol["selection_policy"]
    _require_digest(
        selection["recipe_source"], selection["recipe_source_sha256"]
    )
    reference = protocol["reference_seed"]
    _require_digest(reference["checkpoint"], reference["checkpoint_sha256"])
    teacher = protocol["teacher"]
    _require_digest(teacher["checkpoint"], teacher["checkpoint_sha256"])
    data = protocol["data"]
    _require_digest(
        data["train_metadata"], data["train_metadata_sha256"]
    )
    _require_digest(
        data["validation_metadata"], data["validation_metadata_sha256"]
    )
    for source, digest in protocol["source_files"].items():
        _require_digest(source, digest)

    base_config = yaml.safe_load((ROOT / selection["recipe_source"]).read_text())
    fixed_epoch = int(selection["fixed_epoch"])
    for replication in protocol["replications"]:
        _require_digest(replication["config"], replication["config_sha256"])
        _require_digest(
            replication["init_checkpoint"],
            replication["init_checkpoint_sha256"],
        )
        config = yaml.safe_load((ROOT / replication["config"]).read_text())
        expected = json.loads(json.dumps(base_config))
        base_seed = int(replication["base_seed"])
        expected["project"]["seed"] = int(replication["distillation_rng_seed"])
        expected["training"]["epochs"] = fixed_epoch
        expected["training"]["init_checkpoint"] = replication["init_checkpoint"]
        expected["paths"][
            "experiment_name"
        ] = f"mag_005_seed{base_seed}_fixed_epoch3"
        if config != expected:
            raise ValueError(
                f"{replication['config']} is not an exact fixed-recipe copy"
            )
        if config["training"]["early_stopping"]["enabled"]:
            raise ValueError("Early stopping is prohibited for replication")
    return protocol, _sha256(protocol_path)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    python = Path(sys.executable)
    libraries = [str(python.parent.parent / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    environment["PYTHONPATH"] = "src"
    return environment


def _checkpoint_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def _validate_output(path: Path, fixed_epoch: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != fixed_epoch:
        raise ValueError(f"{path} is not epoch {fixed_epoch}")
    model_config = payload["config"]["model"]
    for field, expected in EXPECTED_MODEL_FIELDS.items():
        if model_config.get(field) != expected:
            raise ValueError(
                f"{path}: {field} expected {expected!r}, "
                f"got {model_config.get(field)!r}"
            )
    model = build_model(model_config)
    model.load_state_dict(payload["model_state_dict"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 808_095:
        raise ValueError(f"{path} has unexpected parameter count {parameters}")
    metrics_path = path.parent / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(path),
        "metrics": str(metrics_path.relative_to(ROOT)),
        "metrics_sha256": _sha256(metrics_path),
        "parameters": parameters,
        "epoch": fixed_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol, protocol_digest = _load_protocol()
    selection = protocol["selection_policy"]
    fixed_epoch = int(selection["fixed_epoch"])
    maximum_batches = int(selection["maximum_train_batches_per_epoch"])

    for replication in protocol["replications"]:
        output = ROOT / replication["output_checkpoint"]
        if output.is_file():
            if _checkpoint_epoch(output) != fixed_epoch:
                raise ValueError(f"Unexpected completed checkpoint: {output}")
            print(f"= completed: {output.relative_to(ROOT)}", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/train_distill.py",
            "--config",
            replication["config"],
            "--device",
            args.device,
            "--max-train-batches",
            str(maximum_batches),
            "--epochs",
            str(fixed_epoch),
        ]
        latest = output.parent / "latest.pt"
        if latest.is_file():
            latest_epoch = _checkpoint_epoch(latest)
            if latest_epoch >= fixed_epoch:
                raise ValueError(
                    f"{latest} reached epoch {latest_epoch} but frozen output "
                    f"{output} is missing"
                )
            command.extend(["--resume", str(latest.relative_to(ROOT))])
        print("+", " ".join(command), flush=True)
        if args.execute:
            subprocess.run(
                command, cwd=ROOT, env=_environment(), check=True
            )

    if not args.execute:
        return
    outputs = [
        _validate_output(ROOT / item["output_checkpoint"], fixed_epoch)
        for item in protocol["replications"]
    ]
    record = {
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_digest,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "fixed_epoch": fixed_epoch,
        "maximum_train_batches_per_epoch": maximum_batches,
        "development_used_for_reselection": False,
        "standard_test_used_during_training": False,
        "outputs": outputs,
    }
    record_path = ROOT / RECORD_PATH
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
