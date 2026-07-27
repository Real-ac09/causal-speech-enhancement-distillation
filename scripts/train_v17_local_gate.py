#!/usr/bin/env python3
"""Train V17 recipe 2 on overlapping two-second oracle windows."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

import train as base
from cnvqg.data import PairedSpeechDataset
from train_v17_gate import (
    _balanced_sampler,
    ordinal_gate_collate_fn,
)


class LocalOrdinalGateDataset(PairedSpeechDataset):
    """Two-second views into fixed four-second V16 parent mixtures."""

    REQUIRED_COLUMNS = {
        "oracle_strength",
        "oracle_class",
        "valid_num_samples",
        "window_start_sample",
        "window_num_samples",
    }
    HOP_LENGTH = 160

    def __init__(self, *args, **kwargs) -> None:
        requested_chunk = kwargs.pop("chunk_seconds", 2.0)
        if requested_chunk is not None and abs(
            float(requested_chunk) - 2.0
        ) > 1e-9:
            raise ValueError("V17 recipe 2 requires two-second windows")
        kwargs["chunk_seconds"] = None
        kwargs["random_crop"] = False
        super().__init__(*args, **kwargs)
        missing = self.REQUIRED_COLUMNS - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"Recipe-2 metadata is missing: {sorted(missing)}"
            )
        if (
            self.metadata["window_num_samples"].astype(int) != 32_000
        ).any():
            raise ValueError("Recipe-2 windows must contain 32,000 samples")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.metadata.iloc[index]
        start = int(row["window_start_sample"])
        end = start + int(row["window_num_samples"])
        noisy = item["noisy"][..., start:end]
        clean = item["clean"][..., start:end]
        required = int(row["window_num_samples"])
        if noisy.shape[-1] < required:
            padding = required - noisy.shape[-1]
            noisy = torch.nn.functional.pad(noisy, (0, padding))
            clean = torch.nn.functional.pad(clean, (0, padding))
        item["noisy"] = noisy
        item["clean"] = clean

        frames = (required + self.HOP_LENGTH - 1) // self.HOP_LENGTH
        frame_rms = clean.reshape(
            1,
            frames,
            self.HOP_LENGTH,
        ).square().mean(dim=-1).sqrt().squeeze(0)
        utterance_rms = clean.square().mean().sqrt()
        speech_threshold = torch.maximum(
            clean.new_tensor(1e-5),
            utterance_rms * 0.01,
        )
        valid_samples = min(int(row["valid_num_samples"]), required)
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
        raise ValueError("Recipe-2 windows forbid clean replacement")
    dataset = LocalOrdinalGateDataset(
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
        collate_fn=ordinal_gate_collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0 and shuffle,
    )


if __name__ == "__main__":
    base.make_loader = make_loader
    base.main()
