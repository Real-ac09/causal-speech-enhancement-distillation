#!/usr/bin/env python3
"""Train V17 with balanced ordinal targets and speech-only frame masking."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import train as base
from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn


class OrdinalGatePairedSpeechDataset(PairedSpeechDataset):
    """Fixed V16 mixtures with V17 class labels and valid-speech masks."""

    REQUIRED_COLUMNS = {
        "oracle_strength",
        "oracle_class",
        "valid_num_samples",
    }
    HOP_LENGTH = 160

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        missing = self.REQUIRED_COLUMNS - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"V17 metadata is missing columns: {sorted(missing)}"
            )
        classes = self.metadata["oracle_class"].astype(int)
        if ((classes < 0) | (classes > 4)).any():
            raise ValueError("V17 oracle classes must lie in [0, 4]")
        valid = self.metadata["valid_num_samples"].astype(int)
        if (valid <= 0).any():
            raise ValueError("V17 valid lengths must be positive")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.metadata.iloc[index]
        clean = item["clean"]
        samples = clean.shape[-1]
        frames = (samples + self.HOP_LENGTH - 1) // self.HOP_LENGTH
        padded = torch.nn.functional.pad(
            clean,
            (0, frames * self.HOP_LENGTH - samples),
        )
        frame_rms = padded.reshape(
            1,
            frames,
            self.HOP_LENGTH,
        ).square().mean(dim=-1).sqrt().squeeze(0)
        utterance_rms = clean.square().mean().sqrt()
        speech_threshold = torch.maximum(
            clean.new_tensor(1e-5),
            utterance_rms * 0.01,
        )
        valid_samples = min(int(row["valid_num_samples"]), samples)
        frame_starts = torch.arange(frames) * self.HOP_LENGTH
        frame_mask = (
            (frame_starts < valid_samples)
            & (frame_rms >= speech_threshold)
        )
        if not bool(frame_mask.any()):
            frame_mask = frame_starts < valid_samples

        item["gate_target_class"] = torch.tensor(
            int(row["oracle_class"]),
            dtype=torch.long,
        )
        item["gate_target_strength"] = torch.tensor(
            float(row["oracle_strength"]),
            dtype=torch.float32,
        )
        item["gate_frame_mask"] = frame_mask
        return item


def ordinal_gate_collate_fn(batch):
    collated = speech_enhancement_collate_fn(batch)
    collated["gate_target_class"] = torch.stack(
        [item["gate_target_class"] for item in batch]
    )
    collated["gate_target_strength"] = torch.stack(
        [item["gate_target_strength"] for item in batch]
    )
    collated["gate_frame_mask"] = torch.stack(
        [item["gate_frame_mask"] for item in batch]
    )
    return collated


def _balanced_sampler(dataset) -> WeightedRandomSampler:
    metadata = dataset.metadata
    domains = metadata["domain"].astype(str)
    classes = metadata["oracle_class"].astype(int)
    domain_counts = domains.value_counts()
    pair_counts = metadata.groupby(
        [domains.rename("domain"), classes.rename("oracle_class")]
    ).size()
    weights = []
    for domain, label in zip(domains, classes):
        domain_count = float(domain_counts[domain])
        class_count = float(pair_counts.loc[(domain, label)])
        # Equal domain mass, with square-root class rebalancing within each
        # domain. This corrects the 80% full-strength skew without repeatedly
        # cloning the single VoiceBank class-zero example.
        weights.append(
            (1.0 / domain_count) * (domain_count / class_count) ** 0.5
        )
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
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
        raise ValueError(
            "V17 labels are tied to fixed mixtures; clean replacement is "
            "not permitted"
        )
    if chunk_seconds is None:
        raise ValueError("V17 recipe 1 requires fixed four-second chunks")
    dataset = OrdinalGatePairedSpeechDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        random_crop=False,
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
        collate_fn=ordinal_gate_collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0 and shuffle,
    )


if __name__ == "__main__":
    base.make_loader = make_loader
    base.main()
