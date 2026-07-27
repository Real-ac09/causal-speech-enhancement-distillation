#!/usr/bin/env python3
"""Train the V15 causal preservation gate with quiet-level augmentation."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

import train as base
from cnvqg.data import speech_enhancement_collate_fn
from train_distill_v15 import QuietLevelPairedSpeechDataset


_ORIGINAL_LOAD_CONFIG = base.load_config
_AUGMENTATION: dict[str, object] = {}


def load_config(path):
    config = _ORIGINAL_LOAD_CONFIG(path)
    data = config["data"]
    _AUGMENTATION.clear()
    _AUGMENTATION.update(
        probability=float(
            data.get("level_augmentation_probability", 0.0)
        ),
        levels=tuple(data.get("target_clean_rms_dbfs", ())),
    )
    return config


def make_loader(
    metadata_csv: str,
    sample_rate: int,
    chunk_seconds: Optional[float],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pcs_target: bool = False,
    clean_input_probability: float = 0.0,
    random_crop: Optional[bool] = None,
) -> DataLoader:
    training = bool(shuffle)
    dataset = QuietLevelPairedSpeechDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        random_crop=(
            shuffle if random_crop is None else bool(random_crop)
        ),
        peak_normalize=False,
        pcs_target=pcs_target,
        clean_input_probability=clean_input_probability,
        target_clean_rms_dbfs=(
            _AUGMENTATION.get("levels", ()) if training else ()
        ),
        level_augmentation_probability=(
            float(_AUGMENTATION.get("probability", 0.0))
            if training
            else 0.0
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=speech_enhancement_collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0 and shuffle,
    )


if __name__ == "__main__":
    base.load_config = load_config
    base.make_loader = make_loader
    base.main()
