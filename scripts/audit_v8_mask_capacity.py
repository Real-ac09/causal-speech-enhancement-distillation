#!/usr/bin/env python3
"""Measure the causal frontend's oracle PESQ under magnitude-ratio bounds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models.factory import build_model


RATIO_CAPS = (1.0, 2.0, math.sqrt(13.0), 4.0, 8.0, 16.0, 32.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = PairedSpeechDataset(
        args.metadata,
        sample_rate=16000,
        chunk_seconds=args.chunk_seconds,
        random_crop=False,
    )
    modes = ["noisy", *(f"ratio_cap_{cap:.6g}" for cap in RATIO_CAPS), "unbounded_ratio"]
    rows: list[dict[str, object]] = []
    ratio_counts = {cap: [] for cap in RATIO_CAPS}
    with torch.inference_mode():
        for index in tqdm(range(min(len(dataset), args.max_items)), desc="mask capacity"):
            item = dataset[index]
            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)
            length = noisy.shape[-1]
            noisy_spectrum, _ = model._analysis(noisy.squeeze(1), pad_end=True)
            clean_spectrum, _ = model._analysis(clean.squeeze(1), pad_end=True)
            noisy_magnitude = noisy_spectrum.abs().clamp_min(1e-7)
            clean_magnitude = clean_spectrum.abs()
            noisy_phase = torch.angle(noisy_spectrum)
            ratio = clean_magnitude / noisy_magnitude
            candidates = {"noisy": model._synthesis(noisy_spectrum, length).unsqueeze(1)}
            for cap in RATIO_CAPS:
                clipped_magnitude = noisy_magnitude * ratio.clamp(max=cap)
                candidates[f"ratio_cap_{cap:.6g}"] = model._synthesis(
                    torch.polar(clipped_magnitude, noisy_phase), length
                ).unsqueeze(1)
                energetic = clean_magnitude > 0.01 * clean_magnitude.amax(
                    dim=(-2, -1), keepdim=True
                )
                ratio_counts[cap].append(float(((ratio > cap) & energetic).float().mean()))
            candidates["unbounded_ratio"] = model._synthesis(
                torch.polar(clean_magnitude, noisy_phase), length
            ).unsqueeze(1)
            noisy_np = noisy.squeeze().float().cpu().numpy()
            clean_np = clean.squeeze().float().cpu().numpy()
            for mode, candidate in candidates.items():
                metrics = compute_speech_metrics(
                    noisy=noisy_np,
                    enhanced=candidate.squeeze().float().cpu().numpy(),
                    clean=clean_np,
                    sample_rate=16000,
                )
                rows.append({"file_id": item["file_id"], "mode": mode, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_file_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {}
    metric_names = [key for key in rows[0] if key not in {"file_id", "mode"}]
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            key: float(np.mean([row[key] for row in selected if row[key] is not None]))
            for key in metric_names
            if any(row[key] is not None for row in selected)
        }
    summary["decoder_geometry"] = {
        "mask_bound": 2.0,
        "maximum_magnitude_ratio": math.sqrt(13.0),
        "energetic_bin_fraction_above_cap": {
            f"{cap:.6g}": float(np.mean(values)) for cap, values in ratio_counts.items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
