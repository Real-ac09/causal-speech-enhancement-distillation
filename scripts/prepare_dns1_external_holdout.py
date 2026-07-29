#!/usr/bin/env python3
"""Validate and lock the paired DNS1 synthetic no-reverb test subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
FILE_ID = re.compile(r"fileid_(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(path: Path) -> int:
    match = FILE_ID.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot extract DNS file ID from {path.name}")
    return int(match.group(1))


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"Audio must be stored beneath {ROOT}: {path}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/external/dns1_interspeech2020"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/dns1_external"),
    )
    args = parser.parse_args()

    subset = (
        args.dataset_root
        / "datasets"
        / "test_set"
        / "synthetic"
        / "no_reverb"
    )
    noisy_directory = subset / "noisy"
    clean_directory = subset / "clean"
    noisy = {_key(path): path for path in sorted(noisy_directory.glob("*.wav"))}
    clean = {_key(path): path for path in sorted(clean_directory.glob("*.wav"))}
    if not noisy or not clean:
        raise FileNotFoundError(
            f"Paired DNS1 audio is missing beneath {subset}; Git LFS pointer "
            "files are not valid audio"
        )
    if set(noisy) != set(clean):
        raise ValueError("DNS1 clean/noisy file IDs do not match")
    if len(noisy) != 150:
        raise ValueError(f"Expected 150 DNS1 no-reverb pairs, found {len(noisy)}")

    rows: list[dict[str, object]] = []
    pair_hashes: list[dict[str, object]] = []
    total_samples = 0
    for file_number in sorted(noisy):
        noisy_path = noisy[file_number]
        clean_path = clean[file_number]
        noisy_info = sf.info(noisy_path)
        clean_info = sf.info(clean_path)
        if (
            noisy_info.samplerate != 16_000
            or clean_info.samplerate != 16_000
            or noisy_info.channels != 1
            or clean_info.channels != 1
        ):
            raise ValueError(f"Unexpected audio format for DNS file {file_number}")
        if noisy_info.frames != clean_info.frames or noisy_info.frames < 1:
            raise ValueError(f"Unaligned DNS pair {file_number}")
        total_samples += noisy_info.frames
        file_id = f"dns1_no_reverb_{file_number:03d}"
        rows.append(
            {
                "split": "external_test",
                "file_id": file_id,
                "speaker_id": "dns1_unknown",
                "noisy_path": _relative(noisy_path),
                "clean_path": _relative(clean_path),
                "sample_rate": 16_000,
                "num_samples": noisy_info.frames,
                "duration_seconds": noisy_info.frames / 16_000,
            }
        )
        pair_hashes.append(
            {
                "file_id": file_id,
                "noisy_sha256": _sha256(noisy_path),
                "clean_sha256": _sha256(clean_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "no_reverb_test.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    manifest = {
        "corpus": "Microsoft DNS Challenge 1 (INTERSPEECH 2020)",
        "condition": "synthetic/no_reverb",
        "source_repository": "https://github.com/microsoft/DNS-Challenge.git",
        "source_branch": "interspeech2020/master",
        "source_commit": "70f19285c36cca4df2338f9248775ddc50980c6b",
        "items": len(rows),
        "sample_rate": 16_000,
        "duration_hours": total_samples / 16_000 / 3600,
        "metadata": _relative(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "pair_hashes": pair_hashes,
        "policy": (
            "Freeze model checkpoints and evaluation policy before first use; "
            "never use this corpus for model or checkpoint selection."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "pair_hashes"}, indent=2))


if __name__ == "__main__":
    main()
