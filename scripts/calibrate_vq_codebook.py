#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn
from cnvqg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an EMA noise codebook.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    with args.config.open() as file:
        config = yaml.safe_load(file)
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    model = build_model(config["model"]).to(device)
    source = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    target_state = model.state_dict()
    compatible = {
        name: value
        for name, value in source["model_state_dict"].items()
        if name in target_state and target_state[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    missing = sorted(set(target_state) - set(compatible))
    if missing != ["vq_gate_logits"]:
        raise RuntimeError(f"Unexpected calibration checkpoint mismatch: {missing}")

    model.noise_vq.update_codebook = True
    model.train()
    dataset = PairedSpeechDataset(
        metadata_csv=config["data"]["train_metadata"],
        project_root=".",
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_seconds=float(config["data"]["chunk_seconds"]),
        random_crop=True,
        peak_normalize=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=speech_enhancement_collate_fn,
        drop_last=True,
    )

    totals = {"perplexity": 0.0, "active": 0.0, "dead": 0.0, "switch": 0.0}
    batches = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="calibrate_vq", total=min(len(loader), args.max_batches)):
            if batches >= args.max_batches:
                break
            noisy = batch["noisy"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                output = model(noisy)
            totals["perplexity"] += float(output.vq.perplexity.detach().cpu())
            totals["active"] += float(output.vq.active_fraction.detach().cpu())
            totals["dead"] += float(output.vq.dead_fraction.detach().cpu())
            totals["switch"] += float(output.vq.switch_rate.detach().cpu())
            batches += 1

    if batches == 0:
        raise RuntimeError("No calibration batches were processed")
    model.noise_vq.update_codebook = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_version": 2,
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "best_val_loss": -float("inf"),
            "config": config,
            "calibration": {
                "source_checkpoint": str(args.source_checkpoint),
                "batches": batches,
                **{name: value / batches for name, value in totals.items()},
                "final_gate_mean": float(
                    torch.sigmoid(model.vq_gate_logits).detach().mean().cpu()
                ),
            },
        },
        temporary,
    )
    temporary.replace(args.output)
    print("Wrote:", args.output)
    print("Calibration:", torch.load(args.output, map_location="cpu", weights_only=False)["calibration"])


if __name__ == "__main__":
    main()
