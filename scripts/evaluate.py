#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models import CNVQGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CN-VQG speech enhancement model.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/cnvqg_baseline/best.pt"),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata/val.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metrics/cnvqg_baseline_val"),
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of utterances for quick tests.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_arg)


def tensor_to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.squeeze().detach().cpu().numpy().astype(np.float32)


def mean_optional(values: List[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]

    if not valid:
        return None

    return float(np.mean(valid))


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = CNVQGModel(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = PairedSpeechDataset(
        metadata_csv=args.metadata,
        project_root=".",
        sample_rate=args.sample_rate,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )

    n_items = len(dataset)

    if args.max_items is not None:
        n_items = min(n_items, args.max_items)

    rows: List[Dict[str, Any]] = []

    print("Checkpoint:", args.checkpoint)
    print("Metadata:", args.metadata)
    print("Output dir:", args.output_dir)
    print("Device:", device)
    print("Uses Mamba:", model.temporal.uses_mamba)
    print("Items:", n_items)

    with torch.no_grad():
        for index in tqdm(range(n_items), desc="Evaluating"):
            item = dataset[index]

            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"]

            output = model(noisy)

            noisy_np = tensor_to_numpy_1d(item["noisy"])
            clean_np = tensor_to_numpy_1d(clean)
            enhanced_np = tensor_to_numpy_1d(output.enhanced.squeeze(0))

            metrics = compute_speech_metrics(
                noisy=noisy_np,
                enhanced=enhanced_np,
                clean=clean_np,
                sample_rate=args.sample_rate,
            )

            row = {
                "file_id": item["file_id"],
                "speaker_id": item["speaker_id"],
                **metrics,
            }

            rows.append(row)

    per_file_csv = args.output_dir / "per_file_metrics.csv"
    summary_json = args.output_dir / "summary.json"

    write_rows_csv(per_file_csv, rows)

    metric_names = [
        key for key in rows[0].keys()
        if key not in {"file_id", "speaker_id"}
    ]

    summary = {
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "num_items": n_items,
        "uses_mamba": model.temporal.uses_mamba,
        "metrics": {
            metric_name: mean_optional([row[metric_name] for row in rows])
            for metric_name in metric_names
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with summary_json.open("w") as file:
        json.dump(summary, file, indent=2)

    print("Wrote:", per_file_csv)
    print("Wrote:", summary_json)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
