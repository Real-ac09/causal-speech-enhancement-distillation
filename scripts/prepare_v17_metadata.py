#!/usr/bin/env python3
"""Prepare V17 ordinal labels and valid-audio lengths from the frozen V16 set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CHUNK_SAMPLES = 64_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strength_class(value: float) -> int:
    distances = [abs(float(value) - level) for level in GRID]
    index = int(min(range(len(GRID)), key=distances.__getitem__))
    if distances[index] > 1e-6:
        raise ValueError(
            f"Oracle strength {value} is not on the frozen V17 grid"
        )
    return index


def _prepare(
    source_path: Path,
    *,
    role: str,
    voice_lengths: dict[str, int],
) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    frame["oracle_class"] = frame["oracle_strength"].map(_strength_class)
    frame["valid_num_samples"] = CHUNK_SAMPLES
    voice = frame["domain"].astype(str) == "voicebank_demand"
    mapped = frame.loc[voice, "source_file_id"].astype(str).map(voice_lengths)
    if mapped.isna().any():
        missing = sorted(
            frame.loc[voice & mapped.isna(), "source_file_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Missing VoiceBank source lengths: {missing[:5]}")
    frame.loc[voice, "valid_num_samples"] = mapped.clip(
        upper=CHUNK_SAMPLES
    ).astype(int)
    frame["v17_role"] = role
    if (frame["valid_num_samples"] <= 0).any():
        raise ValueError("Every V17 item must contain valid audio")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-source",
        type=Path,
        default=Path("results/v16/oracle_labels/train/metadata.csv"),
    )
    parser.add_argument(
        "--calibration-source",
        type=Path,
        default=Path(
            "results/v16/oracle_labels/calibration/metadata.csv"
        ),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/v16/oracle_corpus_selection.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v17_balanced_ordinal"),
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    voice_lengths = {
        str(item["file_id"]): int(item["num_samples"])
        for item in selection["voice_items"]
    }
    train = _prepare(
        args.train_source,
        role="training",
        voice_lengths=voice_lengths,
    )
    calibration = _prepare(
        args.calibration_source,
        role="calibration",
        voice_lengths=voice_lengths,
    )

    train_speakers = set(train["speaker_id"].astype(str))
    calibration_speakers = set(calibration["speaker_id"].astype(str))
    overlap = sorted(train_speakers & calibration_speakers)
    if overlap:
        raise ValueError(
            f"V17 train/calibration speaker overlap: {overlap[:5]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.csv"
    calibration_path = args.output_dir / "calibration.csv"
    train.to_csv(train_path, index=False)
    calibration.to_csv(calibration_path, index=False)

    manifest = {
        "status": "prepared",
        "recipe": "v17_balanced_ordinal_recipe1",
        "strength_grid": list(GRID),
        "chunk_samples": CHUNK_SAMPLES,
        "sources": {
            "train": {
                "path": str(args.train_source),
                "sha256": _sha256(args.train_source),
            },
            "calibration": {
                "path": str(args.calibration_source),
                "sha256": _sha256(args.calibration_source),
            },
            "selection": {
                "path": str(args.selection),
                "sha256": _sha256(args.selection),
            },
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "items": int(len(train)),
            },
            "calibration": {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
                "items": int(len(calibration)),
            },
        },
        "speaker_overlap": overlap,
        "class_counts": {
            split: {
                str(int(label)): int(count)
                for label, count in data["oracle_class"]
                .value_counts()
                .sort_index()
                .items()
            }
            for split, data in (
                ("train", train),
                ("calibration", calibration),
            )
        },
        "domain_class_counts": {
            split: {
                f"{domain}:{int(label)}": int(count)
                for (domain, label), count in data.groupby(
                    ["domain", "oracle_class"]
                ).size().items()
            }
            for split, data in (
                ("train", train),
                ("calibration", calibration),
            )
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
