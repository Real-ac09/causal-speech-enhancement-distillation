#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and lock a speaker-disjoint holdout.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--speaker-column", default="speaker_id")
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    frame = pd.read_csv(args.metadata)
    if args.speaker_column not in frame:
        raise ValueError(f"Missing speaker column: {args.speaker_column}")
    speakers = pd.Series(frame[args.speaker_column].dropna().unique()).sample(
        frac=1.0, random_state=args.seed
    )
    count = max(1, round(len(speakers) * args.fraction))
    held_speakers = set(speakers.iloc[:count])
    holdout = frame[frame[args.speaker_column].isin(held_speakers)].copy()
    development = frame[~frame[args.speaker_column].isin(held_speakers)].copy()
    if set(holdout[args.speaker_column]) & set(development[args.speaker_column]):
        raise RuntimeError("Speaker leakage detected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development_path = args.output_dir / "development.csv"
    holdout_path = args.output_dir / "final_holdout.csv"
    development.to_csv(development_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    manifest = {
        "source": str(args.metadata),
        "source_sha256": sha256(args.metadata),
        "seed": args.seed,
        "speaker_column": args.speaker_column,
        "development_items": len(development),
        "holdout_items": len(holdout),
        "holdout_speakers": sorted(map(str, held_speakers)),
        "holdout_sha256": sha256(holdout_path),
        "policy": "Do not evaluate the final holdout until model selection is frozen.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
