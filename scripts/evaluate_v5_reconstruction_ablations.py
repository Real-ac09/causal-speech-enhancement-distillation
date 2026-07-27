#!/usr/bin/env python3
"""Disentangle V5 magnitude, phase, and frontend effects on speech metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models.factory import build_model


MODES = (
    "enhanced",
    "estimated_magnitude_noisy_phase",
    "noisy_magnitude_predicted_phase",
    "estimated_magnitude_clean_phase",
    "clean_magnitude_predicted_phase",
    "clean_magnitude_noisy_phase",
    "frontend_identity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def waveform(model, magnitude, phase, length: int) -> torch.Tensor:
    return model._synthesis(torch.polar(magnitude, phase), length).unsqueeze(1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    if not all(hasattr(model, name) for name in ("_analysis", "_synthesis")):
        raise TypeError("This diagnostic requires the V5 causal STFT model")

    dataset = PairedSpeechDataset(
        args.metadata,
        sample_rate=16000,
        chunk_seconds=args.chunk_seconds,
        random_crop=False,
    )
    rows: list[dict[str, object]] = []
    count = min(len(dataset), args.max_items)
    with torch.inference_mode():
        for index in tqdm(range(count), desc="V5 reconstruction ablations"):
            item = dataset[index]
            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)
            length = noisy.shape[-1]
            output = model(noisy)
            noisy_spectrum, _ = model._analysis(noisy.squeeze(1), pad_end=True)
            clean_spectrum, _ = model._analysis(clean.squeeze(1), pad_end=True)
            noisy_magnitude, noisy_phase = noisy_spectrum.abs(), torch.angle(noisy_spectrum)
            clean_magnitude, clean_phase = clean_spectrum.abs(), torch.angle(clean_spectrum)
            candidates = {
                "enhanced": output.enhanced,
                "estimated_magnitude_noisy_phase": waveform(
                    model, output.estimated_magnitude, noisy_phase, length
                ),
                "noisy_magnitude_predicted_phase": waveform(
                    model, noisy_magnitude, output.predicted_phase, length
                ),
                "estimated_magnitude_clean_phase": waveform(
                    model, output.estimated_magnitude, clean_phase, length
                ),
                "clean_magnitude_predicted_phase": waveform(
                    model, clean_magnitude, output.predicted_phase, length
                ),
                "clean_magnitude_noisy_phase": waveform(
                    model, clean_magnitude, noisy_phase, length
                ),
                "frontend_identity": waveform(model, noisy_magnitude, noisy_phase, length),
            }
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
    summary = {}
    metric_names = [key for key in rows[0] if key not in {"file_id", "mode"}]
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            key: float(np.mean([row[key] for row in selected if row[key] is not None]))
            for key in metric_names
            if any(row[key] is not None for row in selected)
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
