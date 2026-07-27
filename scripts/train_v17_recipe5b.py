#!/usr/bin/env python3
"""Train Recipe 5b with Recipe-4 balanced domain/class sampling."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

import train as base
from train_v17_gate import _balanced_sampler
from train_v17_recipe5 import (
    UtilitySafetyGateDataset,
    recipe5_collate_fn,
)


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
    if clean_input_probability != 0.0:
        raise ValueError("Recipe 5b forbids clean replacement")
    dataset = UtilitySafetyGateDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        peak_normalize=False,
        pcs_target=pcs_target,
        clean_input_probability=0.0,
    )
    sampler = _balanced_sampler(dataset) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=recipe5_collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0 and shuffle,
    )


if __name__ == "__main__":
    base.make_loader = make_loader
    base.main()
