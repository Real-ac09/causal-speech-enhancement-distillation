#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Check processed speech dataset.")

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata/train.csv"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=4.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = PairedSpeechDataset(
        metadata_csv=args.metadata,
        project_root=".",
        sample_rate=16000,
        chunk_seconds=args.chunk_seconds,
        random_crop=True,
    )

    print(f"Dataset size: {len(dataset)}")

    item = dataset[0]
    print("Single item:")
    print(f"  file_id: {item['file_id']}")
    print(f"  speaker_id: {item['speaker_id']}")
    print(f"  noisy shape: {item['noisy'].shape}")
    print(f"  clean shape: {item['clean'].shape}")
    print(f"  sample_rate: {item['sample_rate']}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=speech_enhancement_collate_fn,
    )

    batch = next(iter(loader))

    print("Batch:")
    print(f"  noisy shape: {batch['noisy'].shape}")
    print(f"  clean shape: {batch['clean'].shape}")
    print(f"  sample_rate: {batch['sample_rate']}")


if __name__ == "__main__":
    main()
