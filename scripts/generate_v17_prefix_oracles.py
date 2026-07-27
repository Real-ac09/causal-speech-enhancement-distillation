#!/usr/bin/env python3
"""Generate explicit one-second causal-prefix targets for V17 Recipe 7b."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cnvqg.metrics import compute_speech_metrics
from generate_v16_oracle_labels import (
    METRICS,
    _annotate_candidates,
    _as_numpy,
    _load_backbone,
    _select_candidate,
    _write_csv_atomic,
    _write_json_atomic,
)
from train_v17_local_gate import LocalOrdinalGateDataset


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
PREFIX_SAMPLES = 16_000
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(
    output_dir: Path,
    candidate_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    progress: dict[str, Any],
) -> None:
    _write_csv_atomic(
        pd.DataFrame(candidate_rows),
        output_dir / "strength_candidates.csv",
    )
    _write_csv_atomic(
        pd.DataFrame(label_rows),
        output_dir / "labels.csv",
    )
    progress.update(
        completed_items=len(label_rows),
        updated_at=_now(),
    )
    _write_json_atomic(progress, output_dir / "progress.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v15/preservation/quiet_level_seed1200/"
            "epoch_003.pt"
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/v17/local_oracle_policy.yaml"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    metadata_path = (ROOT / args.metadata).resolve()
    checkpoint_path = (ROOT / args.backbone_checkpoint).resolve()
    policy_path = (ROOT / args.policy).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    policy = yaml.safe_load(policy_path.read_text())
    if policy["status"] != "frozen":
        raise ValueError("Prefix oracle policy must be frozen")
    strengths = tuple(float(value) for value in policy["strength_grid"])
    if strengths != GRID:
        raise ValueError("Recipe 7b requires the five-level strength grid")

    dataset = LocalOrdinalGateDataset(
        metadata_csv=str(metadata_path),
        project_root=ROOT,
        sample_rate=SAMPLE_RATE,
        chunk_seconds=2.0,
        peak_normalize=False,
        pcs_target=False,
        clean_input_probability=0.0,
    )
    target_items = len(dataset)
    if args.max_items is not None:
        target_items = min(target_items, args.max_items)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "running",
        "started_at": _now(),
        "metadata": str(args.metadata),
        "metadata_sha256": _sha256(metadata_path),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_checkpoint_sha256": _sha256(checkpoint_path),
        "policy": str(args.policy),
        "policy_sha256": _sha256(policy_path),
        "prefix_samples": PREFIX_SAMPLES,
        "prefix_seconds": PREFIX_SAMPLES / SAMPLE_RATE,
        "target_items": target_items,
        "strength_grid": list(strengths),
        "device": args.device,
        "development_set_used": False,
        "external_test_used": False,
    }
    candidates_path = output_dir / "strength_candidates.csv"
    labels_path = output_dir / "labels.csv"
    progress_path = output_dir / "progress.json"
    if args.resume and candidates_path.is_file() and labels_path.is_file():
        previous = json.loads(progress_path.read_text())
        for key in (
            "metadata_sha256",
            "backbone_checkpoint_sha256",
            "policy_sha256",
            "prefix_samples",
            "target_items",
        ):
            if previous.get(key) != progress[key]:
                raise ValueError(f"Resume mismatch for {key}")
        candidate_rows = pd.read_csv(candidates_path).to_dict("records")
        label_rows = pd.read_csv(labels_path).to_dict("records")
        progress["started_at"] = previous["started_at"]
        progress["resumed_at"] = _now()
    elif any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty; pass --resume for a matching run"
        )
    else:
        candidate_rows = []
        label_rows = []
        _write_json_atomic(progress, progress_path)

    completed = {str(row["file_id"]) for row in label_rows}
    model = _load_backbone(checkpoint_path, torch.device(args.device))
    since_checkpoint = 0
    with torch.inference_mode():
        for index in tqdm(range(target_items), desc="V17 prefix oracles"):
            item = dataset[index]
            file_id = str(item["file_id"])
            if file_id in completed:
                continue
            noisy = item["noisy"].unsqueeze(0).to(args.device)
            clean = item["clean"][..., :PREFIX_SAMPLES]
            base = model(noisy).enhanced[..., :PREFIX_SAMPLES]
            noisy_prefix = noisy[..., :PREFIX_SAMPLES]
            noisy_np = _as_numpy(noisy_prefix)
            clean_np = _as_numpy(clean)
            item_rows: list[dict[str, Any]] = []
            for strength in strengths:
                enhanced = noisy_prefix + strength * (
                    base - noisy_prefix
                )
                metrics = compute_speech_metrics(
                    noisy=noisy_np,
                    enhanced=_as_numpy(enhanced),
                    clean=clean_np,
                    sample_rate=SAMPLE_RATE,
                )
                item_rows.append(
                    {
                        "file_id": file_id,
                        "prefix_samples": PREFIX_SAMPLES,
                        "strength": strength,
                        **metrics,
                    }
                )
            item_rows = _annotate_candidates(item_rows, policy)
            selected, reason = _select_candidate(item_rows)
            candidate_rows.extend(item_rows)
            label_rows.append(
                {
                    "file_id": file_id,
                    "prefix_samples": PREFIX_SAMPLES,
                    "prefix_oracle_strength": float(selected["strength"]),
                    "prefix_oracle_class": strengths.index(
                        float(selected["strength"])
                    ),
                    "prefix_oracle_selection_reason": reason,
                    "prefix_oracle_feasible": bool(selected["feasible"]),
                    "prefix_oracle_utility": float(selected["utility"]),
                    "prefix_oracle_constraint_violation_normalised": float(
                        selected["constraint_violation_normalised"]
                    ),
                    **{
                        f"prefix_oracle_{metric}": (
                            float(selected[f"enhanced_{metric}"])
                            if metric in str(
                                selected["available_metrics"]
                            ).split(";")
                            else None
                        )
                        for metric in METRICS
                    },
                }
            )
            completed.add(file_id)
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                _checkpoint(
                    output_dir,
                    candidate_rows,
                    label_rows,
                    progress,
                )
                since_checkpoint = 0

    _checkpoint(output_dir, candidate_rows, label_rows, progress)
    labels = pd.DataFrame(label_rows)
    summary = {
        "status": "complete",
        "items": len(labels),
        "prefix_samples": PREFIX_SAMPLES,
        "prefix_seconds": PREFIX_SAMPLES / SAMPLE_RATE,
        "strength_counts": {
            f"{float(key):.2f}": int(value)
            for key, value in labels["prefix_oracle_strength"]
            .value_counts()
            .sort_index()
            .items()
        },
        "feasible_items": int(labels["prefix_oracle_feasible"].sum()),
        "development_set_used": False,
        "external_test_used": False,
    }
    _write_json_atomic(summary, output_dir / "summary.json")
    progress.update(
        status="complete",
        completed_items=len(labels),
        completed_at=_now(),
    )
    _write_json_atomic(progress, progress_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
