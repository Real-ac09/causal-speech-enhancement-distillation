#!/usr/bin/env python3
"""V15 candidate-B entry point with isolated identity-example sampling."""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

import train_distill as v14
from cnvqg.data import speech_enhancement_collate_fn
from train_distill_v15 import QuietLevelPairedSpeechDataset


class QuietLevelIdentityDataset(QuietLevelPairedSpeechDataset):
    """Add identity examples without perturbing crop or level RNG draws."""

    def __init__(
        self,
        *args,
        identity_probability: float = 0.0,
        **kwargs,
    ) -> None:
        # Identity replacement happens after the parent has made the exact same
        # crop and level decisions as candidate A.
        kwargs["clean_input_probability"] = 0.0
        super().__init__(*args, **kwargs)
        self.identity_probability = float(identity_probability)
        if not 0.0 <= self.identity_probability <= 1.0:
            raise ValueError("identity_probability must be in [0, 1]")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.identity_probability == 0.0:
            return item
        # torch.initial_seed() is worker- and epoch-specific. A local generator
        # prevents this decision from consuming Python's crop/level RNG stream.
        decision_seed = (
            int(torch.initial_seed())
            ^ (int(index) * 0x9E3779B97F4A7C15)
            ^ 0x15B1D3
        ) & ((1 << 64) - 1)
        if random.Random(decision_seed).random() < self.identity_probability:
            item["noisy"] = item["clean"].clone()
        return item


def loader(config: dict, split: str) -> DataLoader:
    training = split == "train"
    data = config["data"]
    dataset = QuietLevelIdentityDataset(
        metadata_csv=data[f"{split}_metadata"],
        sample_rate=int(data["sample_rate"]),
        chunk_seconds=float(data["chunk_seconds"]),
        random_crop=training,
        identity_probability=(
            float(data.get("clean_input_probability", 0.0))
            if training
            else 0.0
        ),
        target_clean_rms_dbfs=(
            tuple(data.get("target_clean_rms_dbfs", ())) if training else ()
        ),
        level_augmentation_probability=(
            float(data.get("level_augmentation_probability", 0.0))
            if training
            else 0.0
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(data["batch_size"]),
        shuffle=training,
        num_workers=int(data["num_workers"]),
        collate_fn=speech_enhancement_collate_fn,
        drop_last=training,
    )


if __name__ == "__main__":
    v14.loader = loader
    v14.main()
