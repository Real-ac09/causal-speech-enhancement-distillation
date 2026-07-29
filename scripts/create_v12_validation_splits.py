#!/usr/bin/env python3
"""Create disjoint internal subsets for the V12 architecture tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _stratified_take(data: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size > len(data):
        raise ValueError(f"Requested {size} rows from a pool of {len(data)}")
    duration_bin = pd.qcut(data["duration_seconds"], q=4, duplicates="drop")
    strata = data["speaker_id"].astype(str) + ":" + duration_bin.astype(str)
    ranked = data.assign(
        _stratum=strata,
        _key=pd.util.hash_pandas_object(data["file_id"], index=False) ^ seed,
    ).sort_values(["_stratum", "_key"])
    ranked["_rank"] = ranked.groupby("_stratum").cumcount()
    return (
        ranked.sort_values(["_rank", "_stratum", "_key"])
        .head(size)
        .drop(columns=["_stratum", "_key", "_rank"])
    )


def _digest(frame: pd.DataFrame) -> str:
    identifiers = "\n".join(sorted(frame["file_id"].astype(str))) + "\n"
    return hashlib.sha256(identifiers.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata/val.csv"),
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=Path(
            "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata"),
    )
    parser.add_argument("--epoch-selection-size", type=int, default=100)
    parser.add_argument("--architecture-selection-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1200)
    args = parser.parse_args()

    source = pd.read_csv(args.source)
    excluded = pd.read_csv(args.exclude)
    if not source["file_id"].is_unique:
        raise ValueError("Source file IDs must be unique")
    pool = source.loc[~source["file_id"].isin(set(excluded["file_id"]))].copy()

    epoch = _stratified_take(pool, args.epoch_selection_size, args.seed)
    pool = pool.loc[~pool["file_id"].isin(set(epoch["file_id"]))].copy()
    architecture = _stratified_take(
        pool, args.architecture_selection_size, args.seed + 1
    )
    reserve = pool.loc[
        ~pool["file_id"].isin(set(architecture["file_id"]))
    ].copy()

    splits = {
        "v12_epoch_selection_100": epoch,
        "v12_architecture_selection_400": architecture,
        "v12_internal_reserve": reserve,
    }
    identifier_sets = {
        name: set(frame["file_id"].astype(str)) for name, frame in splits.items()
    }
    names = list(identifier_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if identifier_sets[first] & identifier_sets[second]:
                raise RuntimeError(f"{first} and {second} overlap")
    excluded_ids = set(excluded["file_id"].astype(str))
    if any(values & excluded_ids for values in identifier_sets.values()):
        raise RuntimeError("A V12 split overlaps the previously reused locked-400 set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": str(args.source),
        "excluded": str(args.exclude),
        "seed": args.seed,
        "source_rows": len(source),
        "excluded_rows": len(excluded),
        "available_rows": sum(len(frame) for frame in splits.values()),
        "splits": {},
    }
    for name, frame in splits.items():
        path = args.output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["splits"][name] = {
            "path": str(path),
            "rows": len(frame),
            "speakers": sorted(frame["speaker_id"].astype(str).unique()),
            "file_id_sha256": _digest(frame),
        }
    manifest_path = args.output_dir / "v12_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
