#!/usr/bin/env python3
"""V15 distillation entry point with opt-in quiet-level augmentation.

The frozen V14 training entry point is intentionally left unchanged. This
wrapper reuses it while replacing only the dataset construction required for
the V15 preservation ablations.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

import train_distill as v14
from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn


class QuietLevelPairedSpeechDataset(PairedSpeechDataset):
    """Randomly rescale a paired example to a configured clean RMS level."""

    def __init__(
        self,
        *args,
        target_clean_rms_dbfs: tuple[float, ...] = (),
        level_augmentation_probability: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_clean_rms_dbfs = tuple(
            float(value) for value in target_clean_rms_dbfs
        )
        self.level_augmentation_probability = float(
            level_augmentation_probability
        )
        if not 0.0 <= self.level_augmentation_probability <= 1.0:
            raise ValueError("level_augmentation_probability must be in [0, 1]")
        if (
            self.level_augmentation_probability > 0.0
            and not self.target_clean_rms_dbfs
        ):
            raise ValueError("Quiet-level augmentation requires target levels")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if (
            self.level_augmentation_probability == 0.0
            or random.random() >= self.level_augmentation_probability
        ):
            return item
        noisy = item["noisy"]
        clean = item["clean"]
        target_dbfs = random.choice(self.target_clean_rms_dbfs)
        clean_rms = clean.square().mean().sqrt().clamp_min(1e-7)
        target_rms = 10.0 ** (target_dbfs / 20.0)
        scale = clean.new_tensor(target_rms) / clean_rms
        joint_peak = torch.maximum(noisy.abs().max(), clean.abs().max())
        if joint_peak * scale > 0.99:
            scale = clean.new_tensor(0.99) / joint_peak.clamp_min(1e-7)
        item["noisy"] = noisy * scale
        item["clean"] = clean * scale
        return item


def loader(config: dict, split: str) -> DataLoader:
    training = split == "train"
    data = config["data"]
    dataset = QuietLevelPairedSpeechDataset(
        metadata_csv=data[f"{split}_metadata"],
        sample_rate=int(data["sample_rate"]),
        chunk_seconds=float(data["chunk_seconds"]),
        random_crop=training,
        clean_input_probability=(
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
