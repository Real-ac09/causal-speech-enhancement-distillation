#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn
from cnvqg.losses import EnhancementLoss
from cnvqg.models import CNVQGModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CN-VQG speech enhancement model.")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_cnvqg.yaml"),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Override config max_train_batches. Useful for smoke tests.",
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Override config max_val_batches. Useful for smoke tests.",
    )

    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_arg)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_loader(
    metadata_csv: str,
    sample_rate: int,
    chunk_seconds: float,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PairedSpeechDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        random_crop=shuffle,
        peak_normalize=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=speech_enhancement_collate_fn,
        drop_last=shuffle,
    )


def save_checkpoint(
    path: Path,
    model: CNVQGModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def append_metrics_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def run_epoch(
    model: CNVQGModel,
    loader: DataLoader,
    criterion: EnhancementLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    mixed_precision: bool = False,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 25,
    max_batches: Optional[int] = None,
    split_name: str = "train",
) -> Dict[str, float]:
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_waveform_l1": 0.0,
        "loss_si_sdr": 0.0,
        "loss_stft": 0.0,
        "loss_vq": 0.0,
        "vq_perplexity": 0.0,
    }

    batches_seen = 0

    progress = tqdm(loader, desc=split_name, leave=False)

    for batch_idx, batch in enumerate(progress, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(
                enabled=mixed_precision and device.type == "cuda"
            ):
                output = model(noisy)
                loss_output = criterion(
                    enhanced=output.enhanced,
                    clean=clean,
                    vq_loss=output.vq.loss,
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None and mixed_precision and device.type == "cuda":
                scaler.scale(loss_output.total).backward()

                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss_output.total.backward()

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                optimizer.step()

        loss_dict = loss_output.as_dict()
        loss_dict["vq_perplexity"] = float(output.vq.perplexity.detach().cpu())

        for key, value in loss_dict.items():
            running[key] += value

        batches_seen += 1

        if batch_idx % log_every == 0:
            progress.set_postfix(
                {
                    "loss": running["loss_total"] / batches_seen,
                    "si_sdr_loss": running["loss_si_sdr"] / batches_seen,
                    "vq_ppl": running["vq_perplexity"] / batches_seen,
                }
            )

    if batches_seen == 0:
        raise RuntimeError(f"No batches were processed for split: {split_name}")

    return {key: value / batches_seen for key, value in running.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config["project"]["seed"]))

    device = get_device(args.device)

    train_loader = make_loader(
        metadata_csv=config["data"]["train_metadata"],
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_seconds=float(config["data"]["chunk_seconds"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        shuffle=True,
    )

    val_loader = make_loader(
        metadata_csv=config["data"]["val_metadata"],
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_seconds=float(config["data"]["chunk_seconds"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        shuffle=False,
    )

    model = CNVQGModel(**config["model"]).to(device)

    criterion = EnhancementLoss(**config["loss"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    mixed_precision = bool(config["training"]["mixed_precision"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=mixed_precision and device.type == "cuda"
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"]) / config["paths"]["experiment_name"]
    metrics_csv = checkpoint_dir / "metrics.csv"

    max_train_batches = args.max_train_batches
    if max_train_batches is None:
        max_train_batches = config["training"].get("max_train_batches")

    max_val_batches = args.max_val_batches
    if max_val_batches is None:
        max_val_batches = config["training"].get("max_val_batches")

    print("Device:", device)
    print("Uses Mamba:", model.temporal.uses_mamba)
    print("Parameter count:", count_parameters(model))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    epochs = int(config["training"]["epochs"])

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
            log_every=int(config["training"]["log_every"]),
            max_batches=max_train_batches,
            split_name="train",
        )

        should_validate = epoch % int(config["training"]["val_every"]) == 0

        if should_validate:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                mixed_precision=False,
                grad_clip_norm=None,
                log_every=int(config["training"]["log_every"]),
                max_batches=max_val_batches,
                split_name="val",
            )
        else:
            val_metrics = {}

        val_loss = val_metrics.get("loss_total", float("inf"))

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }

        append_metrics_row(metrics_csv, row)

        print("Train:", train_metrics)

        if val_metrics:
            print("Val:", val_metrics)

        save_checkpoint(
            path=checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )

            print(f"New best validation loss: {best_val_loss:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
