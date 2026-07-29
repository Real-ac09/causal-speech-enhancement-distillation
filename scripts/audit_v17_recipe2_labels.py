#!/usr/bin/env python3
"""Audit V17 recipe-2 local labels before controller training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _split_summary(frame: pd.DataFrame) -> dict:
    parent_variation = frame.groupby("parent_file_id")[
        "oracle_class"
    ].nunique()
    return {
        "windows": int(len(frame)),
        "parents": int(frame["parent_file_id"].nunique()),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in frame["oracle_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "strength_counts": {
            f"{float(label):.2f}": int(count)
            for label, count in frame["oracle_strength"]
            .value_counts()
            .sort_index()
            .items()
        },
        "domain_class_counts": {
            f"{domain}:{int(label)}": int(count)
            for (domain, label), count in frame.groupby(
                ["domain", "oracle_class"]
            ).size().items()
        },
        "parents_with_multiple_local_classes": int(
            (parent_variation > 1).sum()
        ),
        "parent_local_class_variation_fraction": float(
            (parent_variation > 1).mean()
        ),
        "valid_audio_seconds": {
            "minimum": float(frame["valid_num_samples"].min() / 16_000),
            "mean": float(frame["valid_num_samples"].mean() / 16_000),
            "maximum": float(frame["valid_num_samples"].max() / 16_000),
        },
        "feasible_fraction": float(frame["oracle_feasible"].mean()),
    }


def _validate(frame: pd.DataFrame, name: str) -> list[str]:
    errors = []
    required = {
        "file_id",
        "parent_file_id",
        "speaker_id",
        "domain",
        "window_start_sample",
        "window_num_samples",
        "valid_num_samples",
        "oracle_strength",
        "oracle_class",
        "oracle_feasible",
    }
    missing = required - set(frame.columns)
    if missing:
        return [f"{name} missing columns {sorted(missing)}"]
    if frame["file_id"].duplicated().any():
        errors.append(f"{name} contains duplicate window IDs")
    if (~frame["window_start_sample"].isin([0, 16_000, 32_000])).any():
        errors.append(f"{name} contains invalid window starts")
    if (frame["window_num_samples"].astype(int) != 32_000).any():
        errors.append(f"{name} contains non-two-second windows")
    valid = frame["valid_num_samples"].astype(int)
    if ((valid < 16_000) | (valid > 32_000)).any():
        errors.append(f"{name} contains invalid real-audio lengths")
    classes = frame["oracle_class"].astype(int)
    if not set(classes).issubset(set(range(len(GRID)))):
        errors.append(f"{name} contains invalid classes")
    expected = classes.map(dict(enumerate(GRID)))
    if ((expected - frame["oracle_strength"].astype(float)).abs() > 1e-8).any():
        errors.append(f"{name} class/strength mapping is inconsistent")
    missing_classes = sorted(set(range(len(GRID))) - set(classes))
    if missing_classes:
        errors.append(f"{name} lacks oracle classes {missing_classes}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    calibration = pd.read_csv(args.calibration)
    errors = _validate(train, "train") + _validate(
        calibration,
        "calibration",
    )
    train_speakers = set(train["speaker_id"].astype(str))
    calibration_speakers = set(calibration["speaker_id"].astype(str))
    overlap = sorted(train_speakers & calibration_speakers)
    if overlap:
        errors.append(
            f"Train/calibration speaker overlap: {overlap[:5]}"
        )
    parent_overlap = sorted(
        set(train["parent_file_id"].astype(str))
        & set(calibration["parent_file_id"].astype(str))
    )
    if parent_overlap:
        errors.append(
            f"Train/calibration parent overlap: {parent_overlap[:5]}"
        )

    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "speaker_overlap": overlap,
        "parent_overlap": parent_overlap,
        "train": _split_summary(train),
        "calibration": _split_summary(calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
