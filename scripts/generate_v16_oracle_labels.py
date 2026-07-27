#!/usr/bin/env python3
"""Generate privileged V16 residual-strength labels on training data."""

from __future__ import annotations

import argparse
import hashlib
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
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("pesq", "si_sdr", "stoi", "estoi")
ALLOWED_ROLES = {"training", "training_calibration"}
PROGRESS_NAME = "progress.json"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _checkpoint_progress(
    *,
    output_dir: Path,
    metadata: pd.DataFrame,
    candidate_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    progress: dict[str, Any],
) -> None:
    candidates = pd.DataFrame(candidate_rows)
    labels = pd.DataFrame(label_rows)
    _write_csv_atomic(candidates, output_dir / "strength_candidates.csv")
    _write_csv_atomic(labels, output_dir / "labels.csv")
    if not labels.empty:
        partial_metadata = metadata[
            metadata["file_id"].isin(set(labels["file_id"]))
        ].merge(labels, on="file_id", validate="one_to_one")
        _write_csv_atomic(
            partial_metadata,
            output_dir / "metadata.partial.csv",
        )
    progress["completed_items"] = int(len(labels))
    progress["updated_at"] = _now()
    _write_json_atomic(progress, output_dir / PROGRESS_NAME)


def _load_resume_rows(
    *,
    output_dir: Path,
    strengths: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_path = output_dir / "strength_candidates.csv"
    labels_path = output_dir / "labels.csv"
    if not candidates_path.is_file() or not labels_path.is_file():
        return [], []
    candidates = pd.read_csv(candidates_path)
    labels = pd.read_csv(labels_path)
    if labels.empty:
        return [], []
    if labels["file_id"].duplicated().any():
        raise ValueError("Resume labels contain duplicate file IDs")
    expected_strengths = set(strengths)
    valid_ids = set()
    for file_id, rows in candidates.groupby("file_id"):
        actual = set(rows["strength"].astype(float))
        if len(rows) == len(strengths) and actual == expected_strengths:
            valid_ids.add(file_id)
    labels = labels[labels["file_id"].isin(valid_ids)].copy()
    valid_ids = set(labels["file_id"])
    candidates = candidates[candidates["file_id"].isin(valid_ids)].copy()
    return (
        candidates.to_dict(orient="records"),
        labels.to_dict(orient="records"),
    )


def _load_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _as_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.squeeze().detach().cpu().numpy().astype(np.float32)


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = row[f"enhanced_{metric}"]
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Oracle metric {metric} is unavailable")
    return float(value)


def _available_metrics(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = []
    missing = []
    for metric in METRICS:
        values = [
            row.get(name)
            for row in rows
            for name in (f"noisy_{metric}", f"enhanced_{metric}")
        ]
        valid = [
            value is not None and np.isfinite(float(value))
            for value in values
        ]
        if all(valid):
            available.append(metric)
        elif not any(valid):
            missing.append(metric)
        else:
            partial_policy = (
                policy.get("missing_metric_policy", {}).get(
                    "partial_availability"
                )
            )
            if partial_policy == (
                "exclude_metric_for_entire_item_and_renormalize_"
                "available_utility_weights"
            ):
                missing.append(metric)
            else:
                raise ValueError(
                    f"Oracle metric {metric} is only partially available"
                )
    missing_policy = policy.get("missing_metric_policy")
    if missing:
        if not missing_policy:
            raise ValueError(
                f"Oracle metrics are unavailable: {sorted(missing)}"
            )
        required = set(missing_policy["required_available_metrics"])
        if not required.issubset(available):
            raise ValueError(
                "Missing-metric oracle rule lacks required metrics: "
                f"{sorted(required - set(available))}"
            )
        if (
            missing_policy["all_strengths_unavailable"]
            != "exclude_and_renormalize_available_utility_weights"
        ):
            raise ValueError("Unsupported missing-metric oracle policy")
    return tuple(available), tuple(missing)


def _annotate_candidates(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    full = next(row for row in rows if float(row["strength"]) == 1.0)
    feasibility = policy["feasibility"]
    utility_policy = policy["utility"]["metrics"]
    available_metrics, missing_metrics = _available_metrics(rows, policy)
    available_weight = sum(
        float(utility_policy[metric]["weight"])
        for metric in available_metrics
    )
    if available_weight <= 0.0:
        raise ValueError("Available oracle utility weight must be positive")
    for row in rows:
        floors = {}
        if "pesq" in available_metrics:
            floors["pesq"] = max(
                float(row["noisy_pesq"])
                + float(feasibility["pesq"]["minimum_noisy_delta"]),
                float(full["enhanced_pesq"])
                + float(
                    feasibility["pesq"][
                        "minimum_full_strength_delta"
                    ]
                ),
            )
        if "si_sdr" in available_metrics:
            floors["si_sdr"] = float(row["noisy_si_sdr"]) + float(
                feasibility["si_sdr"]["minimum_noisy_delta_db"]
            )
        if "stoi" in available_metrics:
            floors["stoi"] = float(row["noisy_stoi"]) + float(
                feasibility["stoi"]["minimum_noisy_delta"]
            )
        if "estoi" in available_metrics:
            floors["estoi"] = float(row["noisy_estoi"]) + float(
                feasibility["estoi"]["minimum_noisy_delta"]
            )
        violations = {}
        utility = 0.0
        for metric in METRICS:
            if metric not in available_metrics:
                row[f"{metric}_floor"] = None
                row[f"{metric}_violation_normalised"] = None
                continue
            value = _metric_value(row, metric)
            scale = float(utility_policy[metric]["scale"])
            weight = (
                float(utility_policy[metric]["weight"])
                / available_weight
            )
            noisy = float(row[f"noisy_{metric}"])
            utility += weight * (value - noisy) / scale
            violations[metric] = max(0.0, floors[metric] - value) / scale
            row[f"{metric}_floor"] = floors[metric]
            row[f"{metric}_violation_normalised"] = violations[metric]
        row["feasible"] = not any(value > 0.0 for value in violations.values())
        row["constraint_violation_normalised"] = float(
            sum(violations.values())
        )
        row["utility"] = float(utility)
        row["available_metrics"] = ";".join(available_metrics)
        row["missing_metrics"] = ";".join(missing_metrics)
    return rows


def _select_candidate(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    feasible = [row for row in rows if bool(row["feasible"])]
    if feasible:
        candidates = feasible
        reason = "feasible_maximum_utility"
    else:
        minimum_violation = min(
            float(row["constraint_violation_normalised"])
            for row in rows
        )
        candidates = [
            row
            for row in rows
            if np.isclose(
                float(row["constraint_violation_normalised"]),
                minimum_violation,
                atol=1e-12,
                rtol=0.0,
            )
        ]
        reason = "fallback_minimum_constraint_violation"
    selected = max(
        candidates,
        key=lambda row: (
            float(row["utility"]),
            (
                float(row["enhanced_pesq"])
                if row["enhanced_pesq"] is not None
                and np.isfinite(float(row["enhanced_pesq"]))
                else -float("inf")
            ),
            (
                float(row["enhanced_stoi"])
                if row["enhanced_stoi"] is not None
                and np.isfinite(float(row["enhanced_stoi"]))
                else -float("inf")
            ),
            float(row["strength"]),
        ),
    )
    return selected, reason


def _validate_metadata(frame: pd.DataFrame, metadata_path: Path) -> None:
    if "oracle_role" not in frame.columns:
        raise ValueError(
            "Oracle input metadata must contain an explicit oracle_role"
        )
    roles = set(frame["oracle_role"].astype(str))
    if not roles or not roles.issubset(ALLOWED_ROLES):
        raise ValueError(
            f"Unsupported oracle roles in {metadata_path}: {sorted(roles)}"
        )
    if "oracle_strength" in frame.columns:
        raise ValueError("Input metadata already contains oracle labels")
    required = {
        "file_id",
        "speaker_id",
        "noisy_path",
        "clean_path",
        "sample_rate",
        "num_samples",
        "duration_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Oracle metadata is missing {sorted(missing)}")
    if frame["file_id"].duplicated().any():
        raise ValueError("Oracle input file IDs must be unique")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
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
        default=Path("configs/v16/oracle_strength_policy.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Persist resumable label progress after this many new items.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted label-generation directory.",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")

    metadata_path = _resolve(args.metadata)
    checkpoint_path = _resolve(args.backbone_checkpoint)
    policy_path = _resolve(args.policy)
    output_dir = _resolve(args.output_dir)
    for path in (metadata_path, checkpoint_path, policy_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(metadata_path)
    _validate_metadata(metadata, metadata_path)
    policy = yaml.safe_load(policy_path.read_text())
    if policy["status"] != "frozen":
        raise ValueError("Oracle policy must be frozen before label generation")
    if bool(policy.get("development_set_permitted", True)):
        raise ValueError("V16 oracle policy must forbid development data")
    strengths = tuple(float(value) for value in policy["strength_grid"])
    if strengths != tuple(sorted(set(strengths))):
        raise ValueError("Oracle strength grid must be sorted and unique")
    if strengths[0] != 0.0 or strengths[-1] != 1.0:
        raise ValueError("Oracle strength grid must include endpoints 0 and 1")

    device = torch.device(args.device)
    model = _load_backbone(checkpoint_path, device)
    dataset = PairedSpeechDataset(
        metadata_csv=metadata_path,
        project_root=ROOT,
        sample_rate=16_000,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    item_count = len(dataset)
    if args.max_items is not None:
        item_count = min(item_count, args.max_items)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    progress = {
        "status": "running",
        "started_at": _now(),
        "input_metadata": str(args.metadata),
        "input_metadata_sha256": _sha256(metadata_path),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_checkpoint_sha256": _sha256(checkpoint_path),
        "policy": str(args.policy),
        "policy_sha256": _sha256(policy_path),
        "target_items": int(item_count),
        "strength_grid": list(strengths),
        "device": args.device,
        "external_test_used": False,
        "development_set_used": False,
    }
    existing_outputs = any(
        (output_dir / name).exists()
        for name in (
            PROGRESS_NAME,
            "strength_candidates.csv",
            "labels.csv",
            "metadata.csv",
            "summary.json",
        )
    )
    if existing_outputs and not args.resume:
        raise FileExistsError(
            f"Oracle output already exists in {output_dir}; pass --resume "
            "only for an interrupted matching run"
        )
    if args.resume:
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without {progress_path}"
            )
        previous = json.loads(progress_path.read_text())
        for key in (
            "input_metadata_sha256",
            "backbone_checkpoint_sha256",
            "target_items",
            "strength_grid",
        ):
            if previous.get(key) != progress[key]:
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"{previous.get(key)!r} != {progress[key]!r}"
                )
        if previous.get("policy_sha256") != progress["policy_sha256"]:
            amendment = policy.get("amendment", {})
            accepted = amendment.get("supersedes_policy_sha256")
            if previous.get("policy_sha256") != accepted:
                raise ValueError(
                    "Resume configuration mismatch for policy_sha256: "
                    f"{previous.get('policy_sha256')!r} != "
                    f"{progress['policy_sha256']!r}"
                )
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
    processed_since_checkpoint = 0

    with torch.inference_mode():
        for index in tqdm(range(item_count), desc="V16 oracle labels"):
            item = dataset[index]
            if str(item["file_id"]) in completed_ids:
                continue
            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"]
            base_enhanced = model(noisy).enhanced[..., : noisy.shape[-1]]
            noisy_np = _as_numpy(noisy)
            clean_np = _as_numpy(clean)
            item_rows: list[dict[str, Any]] = []
            for strength in strengths:
                enhanced = noisy + strength * (base_enhanced - noisy)
                metrics = compute_speech_metrics(
                    noisy=noisy_np,
                    enhanced=_as_numpy(enhanced),
                    clean=clean_np,
                    sample_rate=16_000,
                )
                item_rows.append(
                    {
                        "file_id": item["file_id"],
                        "strength": strength,
                        **metrics,
                    }
                )
            item_rows = _annotate_candidates(item_rows, policy)
            selected, reason = _select_candidate(item_rows)
            candidate_rows.extend(item_rows)
            label_rows.append(
                {
                    "file_id": item["file_id"],
                    "oracle_strength": float(selected["strength"]),
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
            completed_ids.add(str(item["file_id"]))
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= args.checkpoint_every:
                _checkpoint_progress(
                    output_dir=output_dir,
                    metadata=metadata.iloc[:item_count],
                    candidate_rows=candidate_rows,
                    label_rows=label_rows,
                    progress=progress,
                )
                processed_since_checkpoint = 0

    labels = pd.DataFrame(label_rows)
    if len(labels) != item_count:
        raise RuntimeError(
            f"Oracle labeling incomplete: {len(labels)}/{item_count} items"
        )
    _checkpoint_progress(
        output_dir=output_dir,
        metadata=metadata.iloc[:item_count],
        candidate_rows=candidate_rows,
        label_rows=label_rows,
        progress=progress,
    )
    labelled_metadata = metadata.iloc[:item_count].merge(
        labels,
        on="file_id",
        validate="one_to_one",
    )
    _write_csv_atomic(labelled_metadata, output_dir / "metadata.csv")
    _write_csv_atomic(labels, output_dir / "labels.csv")
    summary = {
        "status": "complete",
        "role": "privileged_training_target_generation",
        "external_test_used": False,
        "development_set_used": False,
        "input_metadata": str(args.metadata),
        "input_metadata_sha256": _sha256(metadata_path),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_checkpoint_sha256": _sha256(checkpoint_path),
        "policy": str(args.policy),
        "policy_sha256": _sha256(policy_path),
        "device": args.device,
        "items": int(len(labels)),
        "strength_counts": {
            f"{float(key):.6g}": int(value)
            for key, value in labels["oracle_strength"]
            .value_counts()
            .sort_index()
            .items()
        },
        "feasible_items": int(labels["oracle_feasible"].sum()),
        "fallback_items": int((~labels["oracle_feasible"]).sum()),
        "items_with_missing_metrics": int(
            labels["oracle_missing_metrics"].fillna("").ne("").sum()
        ),
        "missing_metric_counts": {
            metric: int(
                labels["oracle_missing_metrics"]
                .fillna("")
                .str.split(";")
                .map(lambda values: metric in values)
                .sum()
            )
            for metric in METRICS
        },
        "claim_policy": policy["claim_policy"],
    }
    _write_json_atomic(summary, output_dir / "summary.json")
    progress.update(
        status="complete",
        completed_items=int(len(labels)),
        completed_at=_now(),
    )
    _write_json_atomic(progress, progress_path)
    (output_dir / "metadata.partial.csv").unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
