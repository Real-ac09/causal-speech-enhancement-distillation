#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn
from cnvqg.losses import EnhancementLoss
from cnvqg.models.factory import build_model
from cnvqg.optimizers import build_optimizer
from cnvqg.training import (
    HybridV2DistillationLoss,
    PrivilegedCausalDistillationLoss,
    V5DistillationLoss,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distil a Hybrid V2 teacher.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model/distillation state from --resume but start a fresh optimizer.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(config: dict, split: str) -> DataLoader:
    training = split == "train"
    dataset = PairedSpeechDataset(
        metadata_csv=config["data"][f"{split}_metadata"],
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_seconds=float(config["data"]["chunk_seconds"]),
        random_crop=training,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=training,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=speech_enhancement_collate_fn,
        drop_last=training,
    )


def atomic_save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run_epoch(
    student,
    teacher,
    clean_loss,
    distillation_loss,
    data_loader,
    device,
    epoch: int,
    warmup_epochs: int,
    optimizer=None,
    maximum_batches=None,
    precision: str = "fp32",
    grad_clip_norm: float = 2.0,
) -> dict[str, float]:
    training = optimizer is not None
    student.train(training)
    distillation_loss.train(training)
    totals: dict[str, float] = {}
    count = 0
    scale = min(1.0, epoch / max(1, warmup_epochs))
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[
        precision
    ]
    mixed_precision = precision != "fp32" and device.type == "cuda"
    for batch_index, batch in enumerate(tqdm(data_loader, leave=False), start=1):
        if maximum_batches is not None and batch_index > maximum_batches:
            break
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            enabled=mixed_precision,
            dtype=autocast_dtype,
        ):
            teacher_output = teacher(noisy)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=mixed_precision,
                dtype=autocast_dtype,
            ):
                student_output = student(noisy)
                if isinstance(distillation_loss, PrivilegedCausalDistillationLoss):
                    distilled = distillation_loss(
                        student_output,
                        teacher_output,
                        clean=clean,
                        noisy=noisy,
                        weight_scale=scale,
                    )
                else:
                    distilled = distillation_loss(
                        student_output,
                        teacher_output,
                        weight_scale=scale,
                    )
            supervised = clean_loss(
                student_output.enhanced.float(),
                clean.float(),
                student_output.vq.loss.float(),
                noisy=noisy.float(),
                noise_prediction=student_output.noise_prediction.float(),
                estimated_noise_magnitude=(
                    getattr(student_output, "noise_spectrum", None).abs().float()
                    if getattr(student_output, "noise_spectrum", None) is not None
                    else None
                ),
                estimated_magnitude=(getattr(student_output, "estimated_magnitude", None).float()
                                     if getattr(student_output, "estimated_magnitude", None) is not None else None),
                magnitude_mask=(getattr(student_output, "magnitude_mask", None).float()
                                if getattr(student_output, "magnitude_mask", None) is not None else None),
                estimated_phase=(getattr(student_output, "estimated_phase", None).float()
                                 if getattr(student_output, "estimated_phase", None) is not None else None),
                phase_candidate=(getattr(student_output, "phase_candidate", None).float()
                                 if getattr(student_output, "phase_candidate", None) is not None else None),
            )
            clean_scale = 0.5 if isinstance(distillation_loss, V5DistillationLoss) else 1.0
            total = clean_scale * supervised.total + distilled.total
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite distillation loss at batch {batch_index}")
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(distillation_loss.parameters()),
                float(grad_clip_norm),
            )
            optimizer.step()
        distilled_values = distilled.detached()
        distilled_values["distilled_total"] = distilled_values.pop("total")
        values = {"total": float(total.detach().cpu()), **distilled_values}
        values["supervised"] = float(supervised.total.detach().cpu())
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    if count == 0:
        raise RuntimeError("No distillation batches processed")
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    args = arguments()
    config = yaml.safe_load(args.config.read_text())
    set_seed(int(config.get("project", {}).get("seed", 42)))
    if args.experiment_name:
        config["paths"]["experiment_name"] = args.experiment_name
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    teacher_checkpoint_path = args.teacher_checkpoint or Path(config["teacher_checkpoint"])
    teacher_checkpoint = torch.load(
        teacher_checkpoint_path, map_location=device, weights_only=False
    )
    teacher = build_model(teacher_checkpoint["config"]["model"]).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval().requires_grad_(False)
    student = build_model(config["student_model"]).to(device)
    initial_checkpoint = config["training"].get("init_checkpoint")
    if initial_checkpoint and args.resume is None:
        initial = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        student.load_state_dict(initial["model_state_dict"])
        print(f"Initialised student from {initial_checkpoint}")
    distillation_type = str(config.get("distillation", {}).get("type", "auto"))
    distillation_arguments = dict(config.get("distillation", {}))
    distillation_arguments.pop("type", None)
    if distillation_type == "privileged_causal":
        distillation = PrivilegedCausalDistillationLoss(
            **distillation_arguments
        ).to(device)
    elif config["student_model"].get("architecture") == "causal_aux_vq_mamba_v5":
        distillation = V5DistillationLoss(
            student_channels=student.channels,
            teacher_channels=teacher.channels,
            student_noise_dim=student.noise_dim,
            teacher_noise_dim=teacher.noise_dim,
            **distillation_arguments,
        ).to(device)
    else:
        distillation = HybridV2DistillationLoss(
            student_speech_dim=student.speech_dim,
            teacher_speech_dim=teacher.speech_dim,
            student_noise_dim=student.noise_dim,
            teacher_noise_dim=teacher.noise_dim,
            **distillation_arguments,
        ).to(device)
    clean_loss = EnhancementLoss(**config["loss"]).to(device)
    optimisation_model = torch.nn.ModuleDict({"student": student, "distillation": distillation})
    optimizer = build_optimizer(optimisation_model, config["training"])
    start_epoch = 1
    best = float("inf")
    precision = str(config["training"].get("precision", "fp32")).lower()
    configured_grad_clip_norm = float(
        config["training"].get("grad_clip_norm", 2.0)
    )
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unknown precision: {precision}")
    checkpoint_metric = str(config["training"].get("checkpoint_metric", "supervised"))
    early_config = config["training"].get("early_stopping", {})
    early_enabled = bool(early_config.get("enabled", False))
    early_patience = int(early_config.get("patience", 8))
    early_min_delta = float(early_config.get("min_delta", 1e-4))
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["student_state_dict"])
        distillation.load_state_dict(checkpoint["distillation_state_dict"])
        if args.reset_optimizer:
            print("Reset optimizer on resume; using learning rates from the current config")
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    train_loader = loader(config, "train")
    val_loader = loader(config, "val")
    run_dir = Path(config["paths"]["checkpoint_dir"]) / config["paths"]["experiment_name"]
    metrics_path = run_dir / "metrics.csv"
    total_epochs = int(args.epochs or config["training"]["epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = run_epoch(
            student, teacher, clean_loss, distillation, train_loader, device,
            epoch, int(config["training"].get("distillation_warmup_epochs", 3)),
            optimizer, args.max_train_batches, precision,
            configured_grad_clip_norm,
        )
        val_metrics = run_epoch(
            student, teacher, clean_loss, distillation, val_loader, device,
            epoch, int(config["training"].get("distillation_warmup_epochs", 3)),
            None, args.max_val_batches, precision,
            configured_grad_clip_norm,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        if checkpoint_metric not in val_metrics:
            raise KeyError(f"Unknown distillation checkpoint metric: {checkpoint_metric}")
        improved = val_metrics[checkpoint_metric] < best - early_min_delta
        if improved:
            best = val_metrics[checkpoint_metric]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        state = {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "distillation_state_dict": distillation.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best,
            "epochs_without_improvement": epochs_without_improvement,
            "config": {"model": config["student_model"]},
            "distillation_config": config,
            "model_state_dict": student.state_dict(),
        }
        atomic_save(run_dir / "latest.pt", state)
        if bool(config["training"].get("save_every_epoch", False)):
            atomic_save(run_dir / f"epoch_{epoch:03d}.pt", state)
        if improved:
            atomic_save(run_dir / "best.pt", state)
        print("epoch", epoch, "train", train_metrics, "val", val_metrics)
        if early_enabled and epochs_without_improvement >= early_patience:
            print(f"Early stopping triggered ({early_patience}/{early_patience}).")
            break


if __name__ == "__main__":
    main()
