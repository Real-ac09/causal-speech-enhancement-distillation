#!/usr/bin/env python3
"""Generate overlapping two-second privileged oracle labels for V17 recipe 2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from generate_v16_oracle_labels import (
    METRICS,
    _annotate_candidates,
    _as_numpy,
    _checkpoint_progress,
    _load_backbone,
    _load_resume_rows,
    _resolve,
    _select_candidate,
    _sha256,
    _validate_metadata,
    _write_csv_atomic,
    _write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 32_000
WINDOW_STARTS = (0, 16_000, 32_000)
MINIMUM_VALID_SAMPLES = 16_000
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PROGRESS_NAME = "progress.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _voice_lengths(selection_path: Path) -> dict[str, int]:
    selection = json.loads(selection_path.read_text())
    return {
        str(item["file_id"]): min(
            64_000,
            int(item["num_samples"]),
        )
        for item in selection["voice_items"]
    }


def _expand_windows(
    metadata: pd.DataFrame,
    voice_lengths: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in metadata.iterrows():
        domain = str(source["domain"])
        if domain == "voicebank_demand":
            source_id = str(source["source_file_id"])
            if source_id not in voice_lengths:
                raise ValueError(
                    f"Missing source length for VoiceBank item {source_id}"
                )
            total_valid = voice_lengths[source_id]
        else:
            total_valid = 64_000
        for window_index, start in enumerate(WINDOW_STARTS):
            valid = min(
                WINDOW_SAMPLES,
                max(0, total_valid - start),
            )
            if valid < MINIMUM_VALID_SAMPLES:
                continue
            row = source.to_dict()
            parent_file_id = str(row["file_id"])
            row.update(
                parent_file_id=parent_file_id,
                file_id=f"{parent_file_id}_local_w{window_index}",
                window_index=window_index,
                window_start_sample=start,
                window_start_seconds=start / SAMPLE_RATE,
                window_num_samples=WINDOW_SAMPLES,
                valid_num_samples=valid,
                num_samples=WINDOW_SAMPLES,
                duration_seconds=WINDOW_SAMPLES / SAMPLE_RATE,
            )
            rows.append(row)
    windows = pd.DataFrame(rows)
    if windows.empty or windows["file_id"].duplicated().any():
        raise ValueError("Expanded local-window metadata is empty or duplicated")
    return windows


def _window_audio(
    item: dict[str, Any],
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    end = start + WINDOW_SAMPLES
    noisy = item["noisy"][..., start:end]
    clean = item["clean"][..., start:end]
    if noisy.shape[-1] < WINDOW_SAMPLES:
        padding = WINDOW_SAMPLES - noisy.shape[-1]
        noisy = torch.nn.functional.pad(noisy, (0, padding))
        clean = torch.nn.functional.pad(clean, (0, padding))
    return noisy, clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/v16/oracle_corpus_selection.json"),
    )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--max-parent-items",
        type=int,
        help="Limit parents before deterministic window expansion (smoke only).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")

    metadata_path = _resolve(args.metadata)
    selection_path = _resolve(args.selection)
    checkpoint_path = _resolve(args.backbone_checkpoint)
    policy_path = _resolve(args.policy)
    output_dir = _resolve(args.output_dir)
    for path in (
        metadata_path,
        selection_path,
        checkpoint_path,
        policy_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent_metadata = pd.read_csv(metadata_path)
    _validate_metadata(parent_metadata, metadata_path)
    if args.max_parent_items is not None:
        parent_metadata = parent_metadata.iloc[
            : args.max_parent_items
        ].copy()
    policy = yaml.safe_load(policy_path.read_text())
    if policy["status"] != "frozen":
        raise ValueError("Local oracle policy must be frozen")
    if bool(policy.get("development_set_permitted", True)):
        raise ValueError("Local oracle policy must forbid development data")
    strengths = tuple(float(value) for value in policy["strength_grid"])
    if strengths != GRID:
        raise ValueError("Recipe 2 requires the frozen five-level grid")

    windows = _expand_windows(
        parent_metadata,
        _voice_lengths(selection_path),
    )
    parent_dataset = PairedSpeechDataset(
        metadata_csv=metadata_path,
        project_root=ROOT,
        sample_rate=SAMPLE_RATE,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    parent_index = {
        str(row["file_id"]): index
        for index, row in parent_dataset.metadata.iterrows()
    }
    device = torch.device(args.device)
    model = _load_backbone(checkpoint_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "running",
        "started_at": _now(),
        "input_metadata": str(args.metadata),
        "input_metadata_sha256": _sha256(metadata_path),
        "selection": str(args.selection),
        "selection_sha256": _sha256(selection_path),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_checkpoint_sha256": _sha256(checkpoint_path),
        "policy": str(args.policy),
        "policy_sha256": _sha256(policy_path),
        "target_parent_items": int(len(parent_metadata)),
        "target_items": int(len(windows)),
        "strength_grid": list(strengths),
        "window_samples": WINDOW_SAMPLES,
        "window_stride_samples": 16_000,
        "minimum_valid_samples": MINIMUM_VALID_SAMPLES,
        "device": args.device,
        "external_test_used": False,
        "development_set_used": False,
    }
    progress_path = output_dir / PROGRESS_NAME
    existing = any(output_dir.iterdir())
    if existing and not args.resume:
        raise FileExistsError(
            f"Local oracle output exists in {output_dir}; use --resume only "
            "for an interrupted matching run"
        )
    if args.resume:
        previous = json.loads(progress_path.read_text())
        for key in (
            "input_metadata_sha256",
            "selection_sha256",
            "backbone_checkpoint_sha256",
            "target_items",
            "strength_grid",
        ):
            if previous.get(key) != progress[key]:
                raise ValueError(f"Resume mismatch for {key}")
        if previous.get("policy_sha256") != progress["policy_sha256"]:
            amendment = policy.get("amendment", {})
            accepted = amendment.get("supersedes_policy_sha256")
            if previous.get("policy_sha256") != accepted:
                raise ValueError("Resume mismatch for policy_sha256")
            progress["resumed_from_policy_sha256"] = accepted
            progress["policy_amendment_id"] = amendment.get("id")
        progress["started_at"] = previous["started_at"]
        progress["resumed_at"] = _now()
        candidate_rows, label_rows = _load_resume_rows(
            output_dir=output_dir,
            strengths=strengths,
        )
    else:
        candidate_rows = []
        label_rows = []
        _write_json_atomic(progress, progress_path)

    completed_ids = {str(row["file_id"]) for row in label_rows}
    since_checkpoint = 0
    with torch.inference_mode():
        for _, window in tqdm(
            windows.iterrows(),
            total=len(windows),
            desc="V17 local oracle labels",
        ):
            window_id = str(window["file_id"])
            if window_id in completed_ids:
                continue
            parent_id = str(window["parent_file_id"])
            item = parent_dataset[parent_index[parent_id]]
            noisy, clean = _window_audio(
                item,
                int(window["window_start_sample"]),
            )
            valid = int(window["valid_num_samples"])
            clean_valid = clean[..., :valid]
            if float(clean_valid.square().mean().sqrt()) < 1e-5:
                raise ValueError(
                    f"Local oracle window has no usable speech: {window_id}"
                )
            noisy_device = noisy.unsqueeze(0).to(device)
            base_enhanced = model(noisy_device).enhanced[
                ..., :WINDOW_SAMPLES
            ]
            noisy_valid = _as_numpy(noisy[..., :valid])
            clean_valid_np = _as_numpy(clean_valid)
            item_rows: list[dict[str, Any]] = []
            for strength in strengths:
                enhanced = noisy_device + strength * (
                    base_enhanced - noisy_device
                )
                metrics = compute_speech_metrics(
                    noisy=noisy_valid,
                    enhanced=_as_numpy(enhanced[..., :valid]),
                    clean=clean_valid_np,
                    sample_rate=SAMPLE_RATE,
                )
                item_rows.append(
                    {
                        "file_id": window_id,
                        "parent_file_id": parent_id,
                        "window_index": int(window["window_index"]),
                        "strength": strength,
                        **metrics,
                    }
                )
            item_rows = _annotate_candidates(item_rows, policy)
            selected, reason = _select_candidate(item_rows)
            candidate_rows.extend(item_rows)
            label_rows.append(
                {
                    "file_id": window_id,
                    "oracle_strength": float(selected["strength"]),
                    "oracle_class": strengths.index(
                        float(selected["strength"])
                    ),
                    "oracle_selection_reason": reason,
                    "oracle_feasible": bool(selected["feasible"]),
                    "oracle_utility": float(selected["utility"]),
                    "oracle_constraint_violation_normalised": float(
                        selected["constraint_violation_normalised"]
                    ),
                    "oracle_available_metrics": selected[
                        "available_metrics"
                    ],
                    "oracle_missing_metrics": selected["missing_metrics"],
                    **{
                        f"oracle_{metric}": (
                            float(selected[f"enhanced_{metric}"])
                            if metric
                            in selected["available_metrics"].split(";")
                            else None
                        )
                        for metric in METRICS
                    },
                }
            )
            completed_ids.add(window_id)
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                _checkpoint_progress(
                    output_dir=output_dir,
                    metadata=windows,
                    candidate_rows=candidate_rows,
                    label_rows=label_rows,
                    progress=progress,
                )
                since_checkpoint = 0

    labels = pd.DataFrame(label_rows)
    if len(labels) != len(windows):
        raise RuntimeError(
            f"Local labeling incomplete: {len(labels)}/{len(windows)}"
        )
    _checkpoint_progress(
        output_dir=output_dir,
        metadata=windows,
        candidate_rows=candidate_rows,
        label_rows=label_rows,
        progress=progress,
    )
    labelled = windows.merge(labels, on="file_id", validate="one_to_one")
    _write_csv_atomic(labelled, output_dir / "metadata.csv")
    _write_csv_atomic(labels, output_dir / "labels.csv")
    summary = {
        "status": "complete",
        "role": "privileged_local_training_target_generation",
        "external_test_used": False,
        "development_set_used": False,
        "parent_items": int(len(parent_metadata)),
        "items": int(len(labels)),
        "strength_counts": {
            f"{float(key):.2f}": int(value)
            for key, value in labels["oracle_strength"]
            .value_counts()
            .sort_index()
            .items()
        },
        "domain_strength_counts": {
            f"{domain}:{float(strength):.2f}": int(count)
            for (domain, strength), count in labelled.groupby(
                ["domain", "oracle_strength"]
            ).size().items()
        },
        "feasible_items": int(labels["oracle_feasible"].sum()),
        "fallback_items": int((~labels["oracle_feasible"]).sum()),
        "items_with_missing_metrics": int(
            labels["oracle_missing_metrics"].fillna("").ne("").sum()
        ),
        "policy": str(args.policy),
        "policy_sha256": _sha256(policy_path),
        "claim_policy": policy["claim_policy"],
    }
    _write_json_atomic(summary, output_dir / "summary.json")
    progress.update(
        status="complete",
        completed_items=int(len(labels)),
        completed_at=_now(),
    )
    _write_json_atomic(progress, progress_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
