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
from cnvqg.models.factory import build_model


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
        "--chunk-seconds",
        type=float,
        default=None,
        help=(
            "Evaluate a deterministic centre crop/padded chunk instead of the "
            "complete utterance. This exists for parity audits; headline "
            "metrics should leave it unset."
        ),
    )

    parser.add_argument(
        "--weights",
        choices=["auto", "model", "ema"],
        default="auto",
        help="Checkpoint weights to evaluate. Auto prefers EMA when available.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument(
        "--output-field",
        choices=["enhanced", "base_enhanced"],
        default="enhanced",
        help="Select the final hybrid output or its waveform base branch.",
    )

    parser.add_argument(
        "--phase-residual-scale",
        type=float,
        default=None,
        help=(
            "Evaluation-only V5.1 phase update scale. Zero uses noisy phase; "
            "one preserves the checkpoint's normal inference path."
        ),
    )

    parser.add_argument(
        "--magnitude-residual-scale",
        type=float,
        default=None,
        help=(
            "Evaluation-only V5.1 log-magnitude update scale. Zero preserves "
            "noisy magnitude; one applies the complete predicted magnitude."
        ),
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="Evaluation-only frontend hop override for a reconstruction audit.",
    )

    parser.add_argument(
        "--refinement-passes",
        type=int,
        default=None,
        help="Evaluation-only tied-refinement depth override.",
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

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]

    model = build_model(config["model"]).to(device)
    ema_state = checkpoint.get("ema_model_state_dict")
    use_ema = args.weights == "ema" or (args.weights == "auto" and ema_state is not None)
    if args.weights == "ema" and ema_state is None:
        raise ValueError("--weights=ema requested but checkpoint contains no EMA weights")
    model.load_state_dict(ema_state if use_ema else checkpoint["model_state_dict"])
    if args.hop_length is not None:
        if args.hop_length < 1 or args.hop_length > int(model.win_length):
            raise ValueError("--hop-length must be between 1 and win_length")
        model.hop_length = int(args.hop_length)
    if args.refinement_passes is not None:
        if not hasattr(model, "refinement_passes"):
            raise ValueError("Model does not expose refinement_passes")
        if args.refinement_passes < 1:
            raise ValueError("--refinement-passes must be positive")
        model.refinement_passes = int(args.refinement_passes)
    if args.phase_residual_scale is not None:
        if not hasattr(model, "phase_residual_scale"):
            raise ValueError(
                "--phase-residual-scale is only supported by models exposing "
                "an evaluation-time phase residual scale"
            )
        if not 0.0 <= args.phase_residual_scale <= 1.0:
            raise ValueError("--phase-residual-scale must be between 0 and 1")
        model.phase_residual_scale = args.phase_residual_scale
    if args.magnitude_residual_scale is not None:
        if not hasattr(model, "magnitude_residual_scale"):
            raise ValueError(
                "--magnitude-residual-scale is only supported by models exposing "
                "an evaluation-time magnitude residual scale"
            )
        if not 0.0 <= args.magnitude_residual_scale <= 1.0:
            raise ValueError("--magnitude-residual-scale must be between 0 and 1")
        model.magnitude_residual_scale = args.magnitude_residual_scale
    model.eval()

    dataset = PairedSpeechDataset(
        metadata_csv=args.metadata,
        project_root=".",
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
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
    uses_mamba = bool(getattr(getattr(model, "temporal", None), "uses_mamba", False))
    print("Uses Mamba:", uses_mamba)
    print("Weights:", "ema" if use_ema else "model")
    print("Chunk seconds:", args.chunk_seconds)
    print("Hop length:", getattr(model, "hop_length", None))
    print("Refinement passes:", getattr(model, "refinement_passes", None))
    if args.phase_residual_scale is not None:
        print("Phase residual scale:", args.phase_residual_scale)
    if args.magnitude_residual_scale is not None:
        print("Magnitude residual scale:", args.magnitude_residual_scale)
    print("Items:", n_items)

    with torch.no_grad():
        for index in tqdm(range(n_items), desc="Evaluating"):
            item = dataset[index]

            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"]

            output = model(noisy)

            noisy_np = tensor_to_numpy_1d(item["noisy"])
            clean_np = tensor_to_numpy_1d(clean)
            selected_output = getattr(output, args.output_field)
            enhanced_np = tensor_to_numpy_1d(selected_output.squeeze(0))

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
        "uses_mamba": uses_mamba,
        "weights": "ema" if use_ema else "model",
        "chunk_seconds": args.chunk_seconds,
        "hop_length": getattr(model, "hop_length", None),
        "refinement_passes": getattr(model, "refinement_passes", None),
        "output_field": args.output_field,
        "phase_residual_scale": args.phase_residual_scale,
        "magnitude_residual_scale": args.magnitude_residual_scale,
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
