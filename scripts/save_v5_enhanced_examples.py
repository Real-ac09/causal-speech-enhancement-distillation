#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from cnvqg.data import PairedSpeechDataset
from cnvqg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Save fixed enhanced examples for quality training/listening.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=400)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = PairedSpeechDataset(args.metadata, chunk_seconds=None, random_crop=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(min(args.max_items, len(dataset))):
            item = dataset[index]
            enhanced = model(item["noisy"].unsqueeze(0).to(device)).enhanced.squeeze().cpu().numpy()
            sf.write(args.output_dir / f"{item['file_id']}.wav", enhanced, 16000)
    print(f"Saved {min(args.max_items, len(dataset))} enhanced examples")


if __name__ == "__main__":
    main()
