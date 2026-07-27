#!/usr/bin/env python3
"""Train the V16 causal gate from precomputed oracle strengths."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import train as base
from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn


class OracleStrengthPairedSpeechDataset(PairedSpeechDataset):
    """Fixed paired audio with one clean-reference oracle label per item."""

    TARGET_COLUMN = "oracle_strength"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.TARGET_COLUMN not in self.metadata.columns:
            raise ValueError(
                f"Oracle metadata requires {self.TARGET_COLUMN!r}"
            )
        strengths = self.metadata[self.TARGET_COLUMN].astype(float)
        if strengths.isna().any():
            raise ValueError("Oracle strengths must not contain missing values")
        if ((strengths < 0.0) | (strengths > 1.0)).any():
            raise ValueError("Oracle strengths must be within [0, 1]")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["gate_target_strength"] = torch.tensor(
            float(self.metadata.iloc[index][self.TARGET_COLUMN]),
            dtype=torch.float32,
        )
        return item


def oracle_strength_collate_fn(batch):
    collated = speech_enhancement_collate_fn(batch)
    collated["gate_target_strength"] = torch.stack(
        [item["gate_target_strength"] for item in batch]
    )
    return collated


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
        raise ValueError(
            "V16 oracle labels are tied to fixed mixtures; clean-input "
            "replacement is not permitted"
        )
    dataset = OracleStrengthPairedSpeechDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        random_crop=False,
        peak_normalize=False,
        pcs_target=pcs_target,
        clean_input_probability=0.0,
    )
    sampler = None
    if shuffle and "domain" in dataset.metadata.columns:
        counts = dataset.metadata["domain"].astype(str).value_counts()
        if len(counts) > 1:
            weights = dataset.metadata["domain"].astype(str).map(
                {domain: 1.0 / count for domain, count in counts.items()}
            )
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights.to_numpy(), dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
            )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=oracle_strength_collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0 and shuffle,
    )


if __name__ == "__main__":
    base.make_loader = make_loader
    base.main()
