#!/usr/bin/env python3
"""Train V17 Recipe 5 with utility, safety, and metric-delta targets."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

import train as base
from train_v17_gate import ordinal_gate_collate_fn
from train_v17_local_gate import LocalOrdinalGateDataset


LEVELS = 5
METRICS = ("pesq", "si_sdr", "stoi", "estoi")


class UtilitySafetyGateDataset(LocalOrdinalGateDataset):
    """Recipe-2 windows augmented with five-candidate sweep supervision."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        required = set()
        for level in range(LEVELS):
            required.update(
                {
                    f"r5_utility_{level}",
                    f"r5_violation_{level}",
                    f"r5_feasible_{level}",
                    f"r5_policy_probability_{level}",
                }
            )
            for metric in METRICS:
                required.update(
                    {
                        f"r5_{metric}_delta_{level}",
                        f"r5_{metric}_available_{level}",
                    }
                )
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"Recipe-5 metadata is missing: {sorted(missing)}"
            )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.metadata.iloc[index]
        item["gate_target_utility"] = torch.tensor(
            [float(row[f"r5_utility_{level}"]) for level in range(LEVELS)],
            dtype=torch.float32,
        )
        item["gate_target_violation"] = torch.tensor(
            [float(row[f"r5_violation_{level}"]) for level in range(LEVELS)],
            dtype=torch.float32,
        )
        item["gate_target_feasible"] = torch.tensor(
            [float(row[f"r5_feasible_{level}"]) for level in range(LEVELS)],
            dtype=torch.float32,
        )
        item["gate_target_policy"] = torch.tensor(
            [
                float(row[f"r5_policy_probability_{level}"])
                for level in range(LEVELS)
            ],
            dtype=torch.float32,
        )
        item["gate_target_metric_deltas"] = torch.tensor(
            [
                [
                    float(row[f"r5_{metric}_delta_{level}"])
                    for metric in METRICS
                ]
                for level in range(LEVELS)
            ],
            dtype=torch.float32,
        )
        item["gate_target_metric_mask"] = torch.tensor(
            [
                [
                    float(row[f"r5_{metric}_available_{level}"])
                    for metric in METRICS
                ]
                for level in range(LEVELS)
            ],
            dtype=torch.float32,
        )
        return item


def recipe5_collate_fn(batch):
    collated = ordinal_gate_collate_fn(batch)
    for key in (
        "gate_target_utility",
        "gate_target_violation",
        "gate_target_feasible",
        "gate_target_policy",
        "gate_target_metric_deltas",
        "gate_target_metric_mask",
    ):
        collated[key] = torch.stack([item[key] for item in batch])
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
        raise ValueError("Recipe 5 forbids clean replacement")
    dataset = UtilitySafetyGateDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        peak_normalize=False,
        pcs_target=pcs_target,
        clean_input_probability=0.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=recipe5_collate_fn,
        drop_last=shuffle,
        persistent_workers=num_workers > 0 and shuffle,
    )


if __name__ == "__main__":
    base.make_loader = make_loader
    base.main()
