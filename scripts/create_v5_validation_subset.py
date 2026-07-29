#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the locked V5 validation subset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if args.size > len(data):
        raise ValueError("Subset size exceeds input size")
    duration_bin = pd.qcut(data["duration_seconds"], q=4, duplicates="drop")
    strata = data["speaker_id"].astype(str) + ":" + duration_bin.astype(str)
    # One stable random key per row, followed by round-robin traversal of
    # speaker/duration strata. This avoids a subset dominated by long files or
    # a small number of speakers.
    ranked = data.assign(
        _stratum=strata,
        _key=pd.util.hash_pandas_object(data["file_id"], index=False) ^ args.seed,
    ).sort_values(["_stratum", "_key"])
    ranked["_rank"] = ranked.groupby("_stratum").cumcount()
    subset = ranked.sort_values(["_rank", "_stratum", "_key"]).head(args.size)
    subset = subset.drop(columns=["_stratum", "_key", "_rank"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)
    print(f"Wrote locked stratified subset: {args.output} ({len(subset)} files)")


if __name__ == "__main__":
    main()
