#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from cnvqg.data import PairedSpeechDataset
from cnvqg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-intrusive SQUIM safeguard for V5.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=400)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    enhancer = build_model(checkpoint["config"]["model"]).to(device)
    enhancer.load_state_dict(checkpoint["model_state_dict"])
    enhancer.eval()
    squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()
    dataset = PairedSpeechDataset(args.metadata, chunk_seconds=None, random_crop=False)
    values = {"squim_stoi": [], "squim_pesq": [], "squim_si_sdr": []}
    with torch.inference_mode():
        for index in range(min(args.max_items, len(dataset))):
            audio = dataset[index]["noisy"].unsqueeze(0).to(device)
            enhanced = enhancer(audio).enhanced.squeeze(1)
            stoi, pesq, si_sdr = squim(enhanced)
            for name, value in zip(values, (stoi, pesq, si_sdr)):
                values[name].append(float(value.squeeze().cpu()))
    report = {name: {"mean": float(np.mean(items)), "std": float(np.std(items))}
              for name, items in values.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
