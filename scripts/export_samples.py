#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from cnvqg.data import PairedSpeechDataset
from cnvqg.models.factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export noisy/clean/enhanced audio samples.")

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
        default=Path("results/audio/cnvqg_baseline_samples"),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=4.0,
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


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = waveform.squeeze().detach().cpu().numpy()
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        file=str(path),
        data=audio,
        samplerate=sample_rate,
        subtype="PCM_16",
    )


def main():
    args = parse_args()
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint["config"]

    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = PairedSpeechDataset(
        metadata_csv=args.metadata,
        project_root=".",
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        random_crop=False,
        peak_normalize=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Checkpoint:", args.checkpoint)
    print("Metadata:", args.metadata)
    print("Output dir:", args.output_dir)
    print("Device:", device)
    print("Uses Mamba:", model.temporal.uses_mamba)

    n = min(args.num_samples, len(dataset))

    with torch.no_grad():
        for index in range(n):
            item = dataset[index]

            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"]
            file_id = item["file_id"]

            output = model(noisy)
            enhanced = output.enhanced.squeeze(0)

            prefix = f"{index:03d}_{file_id}"

            noisy_path = args.output_dir / f"{prefix}_noisy.wav"
            clean_path = args.output_dir / f"{prefix}_clean.wav"
            enhanced_path = args.output_dir / f"{prefix}_enhanced.wav"

            save_audio(noisy_path, item["noisy"], args.sample_rate)
            save_audio(clean_path, clean, args.sample_rate)
            save_audio(enhanced_path, enhanced, args.sample_rate)

            print(f"Exported {prefix}")
            print(f"  noisy:    {noisy_path}")
            print(f"  clean:    {clean_path}")
            print(f"  enhanced: {enhanced_path}")

    print("Sample export complete.")


if __name__ == "__main__":
    main()
