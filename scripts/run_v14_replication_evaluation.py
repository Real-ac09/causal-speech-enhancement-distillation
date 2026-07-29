#!/usr/bin/env python3
"""Execute the frozen V14.2 multi-seed and external-holdout evaluation."""

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
DEFAULT_PROTOCOL = Path("configs/v14/frozen_replication_evaluation.yaml")
RECORD_PATH = Path("results/v14/replication_evaluation/execution_record.json")


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


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        return sum(1 for _ in reader)


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol_path = ROOT / path
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise ValueError("Evaluation protocol must have status: frozen")
    training = protocol["training_execution"]
    _require_digest(training["path"], training["sha256"])
    for source, digest in protocol["source_files"].items():
        _require_digest(source, digest)
    for dataset in protocol["datasets"].values():
        _require_digest(dataset["metadata"], dataset["metadata_sha256"])
        if _csv_rows(ROOT / dataset["metadata"]) != int(dataset["items"]):
            raise ValueError(
                f"Dataset row count changed: {dataset['metadata']}"
            )
        manifest = dataset.get("manifest")
        if manifest is not None:
            _require_digest(manifest, dataset["manifest_sha256"])
    expected_seeds = {1200, 1201, 1202}
    for model_name, model in protocol["models"].items():
        seeds = {int(item["seed"]) for item in model["checkpoints"]}
        if seeds != expected_seeds:
            raise ValueError(
                f"{model_name} expected seeds 1200-1202, got {sorted(seeds)}"
            )
        for checkpoint in model["checkpoints"]:
            _require_digest(checkpoint["path"], checkpoint["sha256"])
            payload = torch.load(
                ROOT / checkpoint["path"],
                map_location="cpu",
                weights_only=False,
            )
            config = payload["config"]["model"]
            model_instance = build_model(config)
            model_instance.load_state_dict(payload["model_state_dict"])
            parameters = sum(
                parameter.numel() for parameter in model_instance.parameters()
            )
            if parameters != int(protocol["model"]["parameter_count"]):
                raise ValueError(
                    f"{checkpoint['path']} has {parameters} parameters"
                )
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


def _run(command: list[str], execute: bool, complete: bool) -> None:
    if complete:
        print("= existing output:", " ".join(command), flush=True)
        return
    print("+", " ".join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def _validate_evaluation(
    directory: Path,
    checkpoint: str,
    metadata: str,
    expected_items: int,
) -> None:
    summary = json.loads((directory / "summary.json").read_text())
    if summary["checkpoint"] != checkpoint:
        raise ValueError(f"{directory} used an unexpected checkpoint")
    if summary["metadata"] != metadata:
        raise ValueError(f"{directory} used unexpected metadata")
    if int(summary["num_items"]) != expected_items:
        raise ValueError(f"{directory} has an unexpected item count")
    if summary["weights"] != "model" or summary["chunk_seconds"] is not None:
        raise ValueError(f"{directory} is not raw-model, full-utterance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--acknowledge-frozen-holdouts",
        action="store_true",
        help="Confirm that neither test corpus can drive model reselection.",
    )
    args = parser.parse_args()
    if args.execute and not args.acknowledge_frozen_holdouts:
        parser.error("--execute requires --acknowledge-frozen-holdouts")
    protocol, protocol_digest = _load_protocol(args.protocol)
    evaluation_root = Path(protocol["evaluation"]["output_root"])
    directories: dict[str, dict[str, list[Path]]] = {}

    for dataset_name, dataset in protocol["datasets"].items():
        directories[dataset_name] = {}
        expected_items = int(dataset["items"])
        for model_name, model in protocol["models"].items():
            model_directories: list[Path] = []
            for checkpoint in model["checkpoints"]:
                seed = int(checkpoint["seed"])
                configured = (
                    protocol.get("existing_evaluations", {})
                    .get(dataset_name, {})
                    .get(model_name, {})
                    .get(str(seed))
                )
                output = (
                    ROOT / configured
                    if configured is not None
                    else ROOT
                    / evaluation_root
                    / dataset_name
                    / model_name
                    / f"seed{seed}"
                )
                model_directories.append(output)
                command = [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--checkpoint",
                    checkpoint["path"],
                    "--metadata",
                    dataset["metadata"],
                    "--output-dir",
                    str(output.relative_to(ROOT)),
                    "--device",
                    args.device,
                    "--weights",
                    "model",
                ]
                _run(
                    command,
                    args.execute,
                    (output / "summary.json").is_file(),
                )
                if args.execute:
                    _validate_evaluation(
                        output,
                        checkpoint["path"],
                        dataset["metadata"],
                        expected_items,
                    )
            directories[dataset_name][model_name] = model_directories

    generated_outputs: list[Path] = []
    bootstrap = protocol["evaluation"]
    for dataset_name, dataset_models in directories.items():
        expected_items = int(protocol["datasets"][dataset_name]["items"])
        for model_name, model_directories in dataset_models.items():
            aggregate = (
                ROOT
                / evaluation_root
                / dataset_name
                / f"{model_name}_aggregate.json"
            )
            command = [
                sys.executable,
                "scripts/aggregate_final_model_seeds.py",
            ]
            for directory in model_directories:
                command.extend(
                    ["--evaluation", str(directory.relative_to(ROOT))]
                )
            command.extend(
                [
                    "--model-name",
                    protocol["models"][model_name]["label"],
                    "--expected-files",
                    str(expected_items),
                    "--bootstrap-samples",
                    str(bootstrap["bootstrap_samples"]),
                    "--bootstrap-seed",
                    str(bootstrap["bootstrap_seed"]),
                    "--output",
                    str(aggregate.relative_to(ROOT)),
                ]
            )
            _run(command, args.execute, aggregate.is_file())
            generated_outputs.append(aggregate)

        comparison = (
            ROOT
            / evaluation_root
            / dataset_name
            / "v14_2_vs_v13_paired_hierarchical.json"
        )
        command = [
            sys.executable,
            "scripts/compare_paired_seed_ensembles.py",
        ]
        for directory in dataset_models["v13"]:
            command.extend(
                ["--reference", str(directory.relative_to(ROOT))]
            )
        for directory in dataset_models["v14_2"]:
            command.extend(
                ["--candidate", str(directory.relative_to(ROOT))]
            )
        command.extend(
            [
                "--reference-name",
                protocol["models"]["v13"]["label"],
                "--candidate-name",
                protocol["models"]["v14_2"]["label"],
                "--expected-files",
                str(expected_items),
                "--bootstrap-samples",
                str(bootstrap["bootstrap_samples"]),
                "--bootstrap-seed",
                str(bootstrap["comparison_bootstrap_seed"]),
                "--output",
                str(comparison.relative_to(ROOT)),
            ]
        )
        _run(command, args.execute, comparison.is_file())
        generated_outputs.append(comparison)

    if not args.execute:
        return
    all_evaluations = {
        directory
        for dataset_models in directories.values()
        for model_directories in dataset_models.values()
        for directory in model_directories
    }
    expected_outputs = [
        *(
            path
            for directory in sorted(all_evaluations)
            for path in (
                directory / "summary.json",
                directory / "per_file_metrics.csv",
            )
        ),
        *generated_outputs,
    ]
    missing = [str(path) for path in expected_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Frozen evaluation outputs are missing: {missing}")
    record = {
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_digest,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "architecture_or_checkpoint_reselection_permitted": False,
        "external_holdout_used_for_selection": False,
        "outputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in expected_outputs
        },
    }
    record_path = ROOT / RECORD_PATH
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
