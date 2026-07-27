#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

from cnvqg.optimizers import build_optimizer, optimizer_lr_summary
from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn
from cnvqg.losses import EnhancementLoss
from cnvqg.metrics import compute_speech_metrics, safe_pesq
from cnvqg.models.factory import build_model


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

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override configured epochs. Useful for smoke tests.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override paths.experiment_name without creating a duplicate config.",
    )
    parser.add_argument(
        "--disable-perceptual-validation",
        action="store_true",
        help="Skip PESQ/STOI validation for short integration smoke tests.",
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


def select_trainable_parameters(model: nn.Module, patterns: list[str]) -> None:
    """Freeze every parameter except names containing an explicit pattern."""
    if not patterns:
        return
    matched = []
    for name, parameter in model.named_parameters():
        enabled = any(pattern in name for pattern in patterns)
        parameter.requires_grad_(enabled)
        if enabled:
            matched.append(name)
    if not matched:
        raise ValueError(
            f"training.trainable_parameter_patterns matched nothing: {patterns}"
        )
    print("Trainable parameter selection:", patterns)
    print("Trainable tensors:", matched)


def load_initial_model_state(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool,
) -> None:
    if strict:
        model.load_state_dict(state_dict)
        return
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in target and target[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    print(
        "Partial initialisation: "
        f"loaded {len(compatible)}/{len(target)} target tensors "
        f"from {len(state_dict)} checkpoint tensors"
    )


class QualityDiscriminator(nn.Module):
    """
    Small training-only quality predictor.

    The legacy mode is a binary clean/non-clean discriminator. The stabilized
    mode regresses normalized PESQ for noisy and enhanced candidates relative
    to the clean reference.
    """

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 15,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels, channels * 2, kernel_size=kernel_size, stride=2, padding=padding),
            nn.GroupNorm(4, channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels * 2, channels * 4, kernel_size=kernel_size, stride=2, padding=padding),
            nn.GroupNorm(8, channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(channels * 4, channels * 4, kernel_size=kernel_size, stride=2, padding=padding),
            nn.GroupNorm(8, channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Linear(channels * 4, 1)

    def forward(self, candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        min_len = min(candidate.shape[-1], reference.shape[-1])
        candidate = candidate[..., :min_len]
        reference = reference[..., :min_len]

        x = torch.cat([candidate, reference], dim=1)
        features = self.net(x).squeeze(-1)
        logits = self.head(features).squeeze(-1)
        return logits


def normalized_pesq_targets(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    target_floor: float = 0.0,
    target_ceiling: float = 1.0,
) -> torch.Tensor:
    """Return MetricGAN-style PESQ MOS targets scaled from [1, 4.5] to [0, 1]."""
    candidate_cpu = candidate.detach().float().cpu()
    reference_cpu = reference.detach().float().cpu()
    targets = []
    for item_index in range(candidate_cpu.shape[0]):
        score = safe_pesq(
            clean=reference_cpu[item_index].squeeze().numpy(),
            estimate=candidate_cpu[item_index].squeeze().numpy(),
            sample_rate=sample_rate,
        )
        if score is None:
            raise RuntimeError("PESQ target computation failed during MetricGAN training")
        normalized = (float(score) - 1.0) / 3.5
        targets.append(min(target_ceiling, max(target_floor, normalized)))
    return candidate.new_tensor(targets, dtype=torch.float32)


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


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
    dataset = PairedSpeechDataset(
        metadata_csv=metadata_csv,
        project_root=".",
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        random_crop=shuffle if random_crop is None else bool(random_crop),
        peak_normalize=False,
        pcs_target=pcs_target,
        clean_input_probability=clean_input_probability,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=speech_enhancement_collate_fn,
        drop_last=shuffle,
        # Validation loaders are consumed before perceptual scoring. Keeping
        # their forked Mamba runtimes alive adds substantial host-memory cost
        # without helping the next phase of the epoch.
        persistent_workers=num_workers > 0 and shuffle,
    )



def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    return ema_model


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()

        if ema_value.dtype.is_floating_point:
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ema_model: Optional[nn.Module] = None,
    epochs_without_improvement: int = 0,
    quality_discriminator: Optional[nn.Module] = None,
    quality_optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    state = {
        "checkpoint_version": 2,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_model_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "quality_discriminator_state_dict": (
            quality_discriminator.state_dict() if quality_discriminator is not None else None
        ),
        "quality_optimizer_state_dict": (
            quality_optimizer.state_dict() if quality_optimizer is not None else None
        ),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": int(epochs_without_improvement),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": config,
    }
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def restore_rng_state(checkpoint: Dict[str, Any]) -> None:
    state = checkpoint.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are loaded with map_location=device. RNG byte states are not
    # model tensors and the CPU generator requires its state on CPU.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def append_metrics_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()

    if path.exists():
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
        added_fields = [name for name in fieldnames if name not in existing_fields]
        if added_fields:
            # Validation can be intentionally intermittent. Expand the schema
            # when its metrics first appear instead of emitting rows wider
            # than the original header.
            fieldnames = existing_fields + added_fields
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
        else:
            fieldnames = existing_fields

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: EnhancementLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    mixed_precision: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 25,
    max_batches: Optional[int] = None,
    split_name: str = "train",
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.999,
    quality_discriminator: Optional[QualityDiscriminator] = None,
    quality_optimizer: Optional[torch.optim.Optimizer] = None,
    metric_weight: float = 0.0,
    quality_grad_clip_norm: Optional[float] = None,
    quality_target_type: str = "binary",
    quality_update_interval: int = 1,
    quality_start_epoch: int = 1,
    quality_stop_epoch: Optional[int] = None,
    metric_generator_start_epoch: int = 1,
    current_epoch: int = 1,
    sample_rate: int = 16000,
    quality_target_floor: float = 0.0,
    quality_target_ceiling: float = 1.0,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    is_train = optimizer is not None
    use_metricgan = (
        is_train
        and quality_discriminator is not None
        and quality_optimizer is not None
        and metric_weight > 0.0
    )
    if quality_target_type not in {"binary", "pesq_regression"}:
        raise ValueError(
            "metricgan_lite.target_type must be 'binary' or 'pesq_regression'"
        )
    if quality_update_interval < 1:
        raise ValueError("metricgan_lite.quality_update_interval must be positive")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    if is_train:
        model.train()
        if quality_discriminator is not None:
            quality_discriminator.train()
    else:
        model.eval()
        if quality_discriminator is not None:
            quality_discriminator.eval()

    running: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_waveform_l1": 0.0,
        "loss_si_sdr": 0.0,
        "loss_stft": 0.0,
        "loss_vq": 0.0,
        "loss_mel": 0.0,
        "loss_complex_stft": 0.0,
        "loss_noise_prediction": 0.0,
        "loss_noise_spectrum": 0.0,
        "loss_magnitude": 0.0,
        "loss_magnitude_log": 0.0,
        "loss_magnitude_ratio": 0.0,
        "loss_phase": 0.0,
        "loss_group_delay": 0.0,
        "loss_instantaneous_frequency": 0.0,
        "loss_phase_confidence": 0.0,
        "loss_compute": 0.0,
        "loss_gate_identity": 0.0,
        "loss_gate_smoothness": 0.0,
        "loss_gate_supervision": 0.0,
        "loss_gate_classification": 0.0,
        "loss_gate_ordinal": 0.0,
        "loss_gate_strength_regression": 0.0,
        "loss_gate_separation": 0.0,
        "loss_gate_utility": 0.0,
        "loss_gate_violation": 0.0,
        "loss_gate_feasibility": 0.0,
        "loss_gate_policy": 0.0,
        "loss_gate_metric_delta": 0.0,
        "loss_metric_g": 0.0,
        "loss_metric_d": 0.0,
        "quality_clean_score": 0.0,
        "quality_enhanced_score": 0.0,
        "quality_noisy_score": 0.0,
        "quality_update_fraction": 0.0,
        "vq_perplexity": 0.0,
        "vq_active_fraction": 0.0,
        "vq_dead_fraction": 0.0,
        "vq_switch_rate": 0.0,
        "vq_gate_mean": 0.0,
        "gate_strength_mean": 0.0,
        "prototype_prediction_loss": 0.0,
        "mixture_consistency_error": 0.0,
    }

    batches_seen = 0
    quality_updates_seen = 0
    metric_generator_batches_seen = 0

    progress = tqdm(loader, desc=split_name, leave=False)
    effective_batches = min(len(loader), max_batches or len(loader))
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(progress, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        metric_g_loss_value = 0.0
        metric_d_loss_value = 0.0
        quality_clean_score = 0.0
        quality_enhanced_score = 0.0
        quality_noisy_score = 0.0

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=mixed_precision and device.type == "cuda",
                dtype=autocast_dtype,
            ):
                output = model(noisy)
            # Loss reductions, logarithms, spectral transforms, and EMA-VQ
            # diagnostics remain FP32 even when the network uses BF16/FP16.
            loss_output = criterion(
                enhanced=output.enhanced.float(),
                clean=clean.float(),
                vq_loss=output.vq.loss.float(),
                noisy=noisy.float(),
                noise_prediction=(
                    getattr(output, "noise_prediction", None).float()
                    if getattr(output, "noise_prediction", None) is not None
                    else None
                ),
                estimated_noise_magnitude=(
                    getattr(output, "noise_spectrum", None).abs().float()
                    if getattr(output, "noise_spectrum", None) is not None
                    else None
                ),
                estimated_magnitude=(
                    getattr(output, "estimated_magnitude", None).float()
                    if getattr(output, "estimated_magnitude", None) is not None
                    else None
                ),
                magnitude_mask=(
                    getattr(output, "magnitude_mask", None).float()
                    if getattr(output, "magnitude_mask", None) is not None
                    else None
                ),
                estimated_phase=(
                    getattr(output, "estimated_phase", None).float()
                    if getattr(output, "estimated_phase", None) is not None
                    else None
                ),
                phase_candidate=(
                    getattr(output, "phase_candidate", None).float()
                    if getattr(output, "phase_candidate", None) is not None
                    else None
                ),
                phase_confidence=(
                    getattr(output, "phase_confidence", None).float()
                    if getattr(output, "phase_confidence", None) is not None
                    else None
                ),
                expected_iterations=getattr(output, "expected_iterations", None),
                gate_strength=getattr(output, "gate_strength", None),
                gate_target_strength=(
                    batch["gate_target_strength"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_strength" in batch
                    else None
                ),
                gate_logits=getattr(output, "gate_logits", None),
                gate_ordinal_logits=getattr(
                    output,
                    "gate_ordinal_logits",
                    None,
                ),
                gate_target_class=(
                    batch["gate_target_class"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_class" in batch
                    else None
                ),
                gate_frame_mask=(
                    batch["gate_frame_mask"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_frame_mask" in batch
                    else None
                ),
                gate_utility=getattr(output, "gate_utility", None),
                gate_log_violation=getattr(
                    output,
                    "gate_log_violation",
                    None,
                ),
                gate_feasibility_logits=getattr(
                    output,
                    "gate_feasibility_logits",
                    None,
                ),
                gate_metric_deltas=getattr(
                    output,
                    "gate_metric_deltas",
                    None,
                ),
                gate_target_utility=(
                    batch["gate_target_utility"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_utility" in batch
                    else None
                ),
                gate_target_violation=(
                    batch["gate_target_violation"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_violation" in batch
                    else None
                ),
                gate_target_feasible=(
                    batch["gate_target_feasible"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_feasible" in batch
                    else None
                ),
                gate_target_policy=(
                    batch["gate_target_policy"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_policy" in batch
                    else None
                ),
                gate_target_metric_deltas=(
                    batch["gate_target_metric_deltas"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_metric_deltas" in batch
                    else None
                ),
                gate_target_metric_mask=(
                    batch["gate_target_metric_mask"].to(
                        device,
                        non_blocking=True,
                    )
                    if "gate_target_metric_mask" in batch
                    else None
                ),
            )

            total_loss = loss_output.total

            if not torch.isfinite(total_loss):
                diagnostics = loss_output.as_dict()
                diagnostics["batch"] = batch_idx
                diagnostics["vq_perplexity"] = float(
                    output.vq.perplexity.detach().float().cpu()
                )
                diagnostics["vq_active_fraction"] = float(
                    getattr(
                        output.vq,
                        "active_fraction",
                        output.vq.perplexity.new_tensor(0.0),
                    ).detach().float().cpu()
                )
                diagnostics["vq_dead_fraction"] = float(
                    getattr(
                        output.vq,
                        "dead_fraction",
                        output.vq.perplexity.new_tensor(0.0),
                    ).detach().float().cpu()
                )
                raise FloatingPointError(
                    f"Non-finite loss in {split_name}: {diagnostics}"
                )

            quality_epoch_active = (
                current_epoch >= quality_start_epoch
                and (quality_stop_epoch is None or current_epoch <= quality_stop_epoch)
            )
            update_quality = (
                use_metricgan
                and quality_epoch_active
                and (batch_idx - 1) % quality_update_interval == 0
            )
            train_with_metric = (
                use_metricgan and current_epoch >= metric_generator_start_epoch
            )

            if update_quality:
                quality_updates_seen += 1
                # 1) Train quality discriminator
                set_requires_grad(quality_discriminator, True)
                quality_optimizer.zero_grad(set_to_none=True)

                enhanced_detached = output.enhanced.detach()

                with torch.cuda.amp.autocast(enabled=False):
                    clean_float = clean.float()
                    noisy_float = noisy.float()
                    enhanced_detached_float = enhanced_detached.float()

                    logits_clean = quality_discriminator(clean_float, clean_float)
                    logits_enhanced = quality_discriminator(enhanced_detached_float, clean_float)
                    logits_noisy = quality_discriminator(noisy_float, clean_float)

                    if quality_target_type == "pesq_regression":
                        clean_targets = torch.full_like(
                            logits_clean, quality_target_ceiling
                        )
                        enhanced_targets = normalized_pesq_targets(
                            enhanced_detached_float,
                            clean_float,
                            sample_rate,
                            quality_target_floor,
                            quality_target_ceiling,
                        )
                        noisy_targets = normalized_pesq_targets(
                            noisy_float,
                            clean_float,
                            sample_rate,
                            quality_target_floor,
                            quality_target_ceiling,
                        )
                        d_loss_clean = F.mse_loss(
                            torch.sigmoid(logits_clean), clean_targets
                        )
                        d_loss_enhanced = F.mse_loss(
                            torch.sigmoid(logits_enhanced), enhanced_targets
                        )
                        d_loss_noisy = F.mse_loss(
                            torch.sigmoid(logits_noisy), noisy_targets
                        )
                    else:
                        ones = torch.ones_like(logits_clean)
                        zeros_enhanced = torch.zeros_like(logits_enhanced)
                        zeros_noisy = torch.zeros_like(logits_noisy)
                        d_loss_clean = F.binary_cross_entropy_with_logits(
                            logits_clean, ones
                        )
                        d_loss_enhanced = F.binary_cross_entropy_with_logits(
                            logits_enhanced, zeros_enhanced
                        )
                        d_loss_noisy = F.binary_cross_entropy_with_logits(
                            logits_noisy, zeros_noisy
                        )

                    metric_d_loss = d_loss_clean + 0.5 * d_loss_enhanced + 0.5 * d_loss_noisy

                metric_d_loss.backward()

                if quality_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        quality_discriminator.parameters(),
                        quality_grad_clip_norm,
                    )

                quality_optimizer.step()

                quality_clean_score = float(torch.sigmoid(logits_clean).mean().detach().cpu())
                quality_enhanced_score = float(torch.sigmoid(logits_enhanced).mean().detach().cpu())
                quality_noisy_score = float(torch.sigmoid(logits_noisy).mean().detach().cpu())
                metric_d_loss_value = float(metric_d_loss.detach().cpu())

            if train_with_metric:
                metric_generator_batches_seen += 1
                # Train the enhancer against the current (possibly frozen)
                # quality regressor. Sparse discriminator updates prevent it
                # from outrunning the enhancer.
                set_requires_grad(quality_discriminator, False)

                with torch.cuda.amp.autocast(enabled=False):
                    logits_enhanced_for_g = quality_discriminator(output.enhanced.float(), clean.float())
                    target_high_quality = torch.full_like(
                        logits_enhanced_for_g, quality_target_ceiling
                    )
                    if quality_target_type == "pesq_regression":
                        metric_g_loss = F.mse_loss(
                            torch.sigmoid(logits_enhanced_for_g),
                            target_high_quality,
                        )
                    else:
                        metric_g_loss = F.binary_cross_entropy_with_logits(
                            logits_enhanced_for_g,
                            target_high_quality,
                        )
                    quality_enhanced_score = float(
                        torch.sigmoid(logits_enhanced_for_g).mean().detach().cpu()
                    )

                total_loss = loss_output.total + float(metric_weight) * metric_g_loss
                metric_g_loss_value = float(metric_g_loss.detach().cpu())

        if is_train:
            backward_loss = total_loss / float(gradient_accumulation_steps)
            should_step = (
                batch_idx % gradient_accumulation_steps == 0
                or batch_idx == effective_batches
            )

            if scaler is not None and mixed_precision and device.type == "cuda":
                scaler.scale(backward_loss).backward()

                if should_step:
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if ema_model is not None:
                        update_ema_model(ema_model, model, ema_decay)
            else:
                backward_loss.backward()

                if should_step:
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if ema_model is not None:
                        update_ema_model(ema_model, model, ema_decay)

            if use_metricgan:
                set_requires_grad(quality_discriminator, True)

        loss_dict = loss_output.as_dict()
        loss_dict["loss_total"] = float(total_loss.detach().cpu())
        loss_dict["loss_metric_g"] = metric_g_loss_value
        loss_dict["loss_metric_d"] = metric_d_loss_value
        loss_dict["quality_clean_score"] = quality_clean_score
        loss_dict["quality_enhanced_score"] = quality_enhanced_score
        loss_dict["quality_noisy_score"] = quality_noisy_score
        loss_dict["quality_update_fraction"] = float(update_quality)
        loss_dict["vq_perplexity"] = float(output.vq.perplexity.detach().cpu())
        loss_dict["vq_active_fraction"] = float(
            getattr(output.vq, "active_fraction", output.vq.perplexity.new_tensor(0.0))
            .detach()
            .cpu()
        )
        loss_dict["vq_dead_fraction"] = float(
            getattr(output.vq, "dead_fraction", output.vq.perplexity.new_tensor(0.0))
            .detach()
            .cpu()
        )
        loss_dict["vq_switch_rate"] = float(
            getattr(output.vq, "switch_rate", output.vq.perplexity.new_tensor(0.0))
            .detach()
            .cpu()
        )
        loss_dict["vq_gate_mean"] = float(
            getattr(output, "vq_gate", output.vq.perplexity.new_tensor(0.0))
            .detach()
            .float()
            .mean()
            .cpu()
        )
        loss_dict["gate_strength_mean"] = float(
            getattr(
                output,
                "gate_strength",
                output.vq.perplexity.new_tensor(1.0),
            )
            .detach()
            .float()
            .mean()
            .cpu()
        )
        loss_dict["prototype_prediction_loss"] = float(
            getattr(
                output,
                "prototype_prediction_loss",
                output.vq.perplexity.new_tensor(0.0),
            ).detach().float().cpu()
        )
        mixture_residual = getattr(output, "mixture_residual", None)
        loss_dict["mixture_consistency_error"] = (
            float(mixture_residual.detach().abs().max().float().cpu())
            if mixture_residual is not None
            else 0.0
        )

        for key, value in loss_dict.items():
            running[key] += value

        batches_seen += 1

        if batch_idx % log_every == 0:
            progress.set_postfix(
                {
                    "loss": running["loss_total"] / batches_seen,
                    "si_sdr_loss": running["loss_si_sdr"] / batches_seen,
                    "metric_g": running["loss_metric_g"] / batches_seen,
                    "metric_d": running["loss_metric_d"] / batches_seen,
                    "vq_ppl": running["vq_perplexity"] / batches_seen,
                }
            )

    if batches_seen == 0:
        raise RuntimeError(f"No batches were processed for split: {split_name}")

    averaged = {key: value / batches_seen for key, value in running.items()}
    if quality_updates_seen:
        for key in ("loss_metric_d", "quality_clean_score", "quality_noisy_score"):
            averaged[key] = running[key] / quality_updates_seen
    if metric_generator_batches_seen:
        for key in ("loss_metric_g", "quality_enhanced_score"):
            averaged[key] = running[key] / metric_generator_batches_seen
    return averaged


@torch.inference_mode()
def run_perceptual_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sample_rate: int,
    max_items: int,
    phase_residual_scale: Optional[float] = None,
    magnitude_residual_scale: Optional[float] = None,
) -> Dict[str, float]:
    if max_items < 1:
        raise ValueError("perceptual_validation.max_items must be positive")
    was_training = model.training
    original_scales: Dict[str, float] = {}
    requested_scales = {
        "phase_residual_scale": phase_residual_scale,
        "magnitude_residual_scale": magnitude_residual_scale,
    }
    for attribute, value in requested_scales.items():
        if value is None:
            continue
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"perceptual_validation.{attribute} must be between 0 and 1")
        if not hasattr(model, attribute):
            raise ValueError(
                f"perceptual_validation.{attribute} is unsupported by this model"
            )
        original_scales[attribute] = float(getattr(model, attribute))
        setattr(model, attribute, value)
    model.eval()
    collected: Dict[str, list[float]] = {}
    items_seen = 0
    try:
        for batch in tqdm(loader, desc="perceptual", leave=False):
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"]
            enhanced = model(noisy).enhanced.detach().float().cpu()
            noisy_cpu = noisy.detach().float().cpu()
            for item_index in range(noisy.shape[0]):
                if items_seen >= max_items:
                    break
                metrics = compute_speech_metrics(
                    noisy=noisy_cpu[item_index].squeeze().numpy(),
                    enhanced=enhanced[item_index].squeeze().numpy(),
                    clean=clean[item_index].squeeze().numpy(),
                    sample_rate=sample_rate,
                )
                for name, value in metrics.items():
                    if value is not None:
                        collected.setdefault(name, []).append(float(value))
                items_seen += 1
            if items_seen >= max_items:
                break
    finally:
        for attribute, value in original_scales.items():
            setattr(model, attribute, value)
        if was_training:
            model.train()
    return {
        f"perceptual_{name}": float(np.mean(values))
        for name, values in collected.items()
        if values
    }



def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.disable_perceptual_validation:
        smoke_training = config.setdefault("training", {})
        smoke_training.setdefault(
            "perceptual_validation", {}
        )["enabled"] = False
        smoke_training["checkpoint_metric"] = "loss_total"
        smoke_training["checkpoint_mode"] = "min"
        scheduler = smoke_training.get("lr_scheduler", {})
        if scheduler.get("name") == "reduce_on_plateau":
            scheduler["metric"] = "loss_total"
            scheduler["mode"] = "min"
    if args.experiment_name is not None:
        config["paths"]["experiment_name"] = args.experiment_name

    set_seed(int(config["project"]["seed"]))

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    perceptual_config = config["training"].get("perceptual_validation", {})
    perceptual_enabled = bool(perceptual_config.get("enabled", False))
    perceptual_whole_utterance = bool(
        perceptual_config.get("whole_utterance", False)
    )

    train_loader = make_loader(
        metadata_csv=config["data"]["train_metadata"],
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_seconds=float(config["data"]["chunk_seconds"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        shuffle=True,
        pcs_target=bool(config["data"].get("pcs_target", False)),
        clean_input_probability=float(
            config["data"].get("clean_input_probability", 0.0)
        ),
        random_crop=bool(config["data"].get("train_random_crop", True)),
    )

    val_loader = make_loader(
        metadata_csv=config["data"]["val_metadata"],
        sample_rate=int(config["data"]["sample_rate"]),
        chunk_seconds=float(config["data"]["chunk_seconds"]),
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        shuffle=False,
    )
    perceptual_val_loader = val_loader
    if perceptual_enabled and perceptual_whole_utterance:
        perceptual_val_loader = make_loader(
            metadata_csv=config["data"]["val_metadata"],
            sample_rate=int(config["data"]["sample_rate"]),
            chunk_seconds=None,
            batch_size=1,
            # A second persistent worker pool duplicates the imported Mamba
            # runtime and can exhaust host RAM. Whole-utterance validation is
            # sequential and dominated by model/metric work, so load it in
            # the parent process.
            num_workers=0,
            shuffle=False,
        )
    full_val_metadata = config["data"].get("full_val_metadata")
    full_val_loader = None
    full_perceptual_val_loader = None
    if full_val_metadata:
        full_val_loader = make_loader(
            metadata_csv=full_val_metadata,
            sample_rate=int(config["data"]["sample_rate"]),
            chunk_seconds=float(config["data"]["chunk_seconds"]),
            batch_size=int(config["data"]["batch_size"]),
            num_workers=int(config["data"]["num_workers"]),
            shuffle=False,
        )
        full_perceptual_val_loader = full_val_loader
        if perceptual_enabled and perceptual_whole_utterance:
            full_perceptual_val_loader = make_loader(
                metadata_csv=full_val_metadata,
                sample_rate=int(config["data"]["sample_rate"]),
                chunk_seconds=None,
                batch_size=1,
                num_workers=0,
                shuffle=False,
            )

    model = build_model(config["model"]).to(device)
    select_trainable_parameters(
        model,
        list(config["training"].get("trainable_parameter_patterns", [])),
    )

    ema_config = config["training"].get("ema", {})
    ema_enabled = bool(ema_config.get("enabled", False))
    ema_decay = float(ema_config.get("decay", 0.999))
    ema_model = create_ema_model(model) if ema_enabled else None

    criterion = EnhancementLoss(**config["loss"])

    optimizer = build_optimizer(model, config["training"])

    configured_precision = config["training"].get("precision")
    if configured_precision is None:
        configured_precision = (
            "fp16" if bool(config["training"].get("mixed_precision", False)) else "fp32"
        )
    configured_precision = str(configured_precision).lower()
    precision_dtypes = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if configured_precision not in precision_dtypes:
        raise ValueError(f"Unknown training precision: {configured_precision}")
    autocast_dtype = precision_dtypes[configured_precision]
    mixed_precision = configured_precision != "fp32"
    scaler = torch.cuda.amp.GradScaler(
        enabled=configured_precision == "fp16" and device.type == "cuda"
    )

    metricgan_config = config["training"].get("metricgan_lite", {})
    metricgan_enabled = bool(metricgan_config.get("enabled", False))
    metric_weight = float(metricgan_config.get("metric_weight", 0.0))
    quality_grad_clip_norm = metricgan_config.get("quality_grad_clip_norm")
    quality_target_type = str(metricgan_config.get("target_type", "binary"))
    quality_update_interval = int(metricgan_config.get("quality_update_interval", 1))
    quality_start_epoch = int(metricgan_config.get("quality_start_epoch", 1))
    quality_stop_epoch_value = metricgan_config.get("quality_stop_epoch")
    quality_stop_epoch = (
        int(quality_stop_epoch_value) if quality_stop_epoch_value is not None else None
    )
    metric_generator_start_epoch = int(
        metricgan_config.get("generator_start_epoch", 1)
    )
    quality_target_floor = float(metricgan_config.get("target_floor", 0.0))
    quality_target_ceiling = float(metricgan_config.get("target_ceiling", 1.0))

    quality_discriminator = None
    quality_optimizer = None

    if metricgan_enabled:
        quality_discriminator = QualityDiscriminator(
            channels=int(metricgan_config.get("channels", 32)),
            kernel_size=int(metricgan_config.get("kernel_size", 15)),
            dropout=float(metricgan_config.get("dropout", 0.10)),
        ).to(device)

        quality_optimizer = torch.optim.AdamW(
            quality_discriminator.parameters(),
            lr=float(metricgan_config.get("quality_learning_rate", 0.0001)),
            weight_decay=float(metricgan_config.get("quality_weight_decay", 0.0001)),
        )

        pretrained_quality = metricgan_config.get("pretrained_checkpoint")
        if pretrained_quality is not None:
            quality_checkpoint = torch.load(
                pretrained_quality, map_location=device, weights_only=False
            )
            calibration = quality_checkpoint.get("calibration", {})
            pearson = float(calibration.get("pearson", -1.0))
            mae = float(calibration.get("mae", float("inf")))
            minimum_pearson = float(metricgan_config.get("minimum_pearson", 0.80))
            maximum_mae = float(metricgan_config.get("maximum_mae", 0.12))
            if pearson < minimum_pearson or mae > maximum_mae:
                raise RuntimeError(
                    "Quality regressor failed calibration gate: "
                    f"Pearson={pearson:.4f} (need {minimum_pearson:.4f}), "
                    f"MAE={mae:.4f} (need <= {maximum_mae:.4f})"
                )
            quality_discriminator.load_state_dict(quality_checkpoint["model_state_dict"])
            print(f"Loaded calibrated quality regressor: Pearson={pearson:.4f}, MAE={mae:.4f}")

        if quality_grad_clip_norm is not None:
            quality_grad_clip_norm = float(quality_grad_clip_norm)

    lr_scheduler_config = config["training"].get("lr_scheduler", {})
    scheduler = None

    if lr_scheduler_config.get("name") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(lr_scheduler_config.get("mode", "min")),
            factor=float(lr_scheduler_config.get("factor", 0.5)),
            patience=int(lr_scheduler_config.get("patience", 3)),
            min_lr=float(lr_scheduler_config.get("min_lr", 1e-5)),
        )
    elif lr_scheduler_config.get("name") == "warmup_cosine":
        total_epochs = int(
            args.epochs if args.epochs is not None else config["training"]["epochs"]
        )
        configured_warmup_epochs = int(lr_scheduler_config.get("warmup_epochs", 3))
        warmup_epochs = min(configured_warmup_epochs, total_epochs)
        minimum_factor = float(lr_scheduler_config.get("minimum_factor", 0.05))
        if configured_warmup_epochs < 0:
            raise ValueError("warmup_cosine requires warmup_epochs >= 0")
        if not 0.0 <= minimum_factor <= 1.0:
            raise ValueError("warmup_cosine minimum_factor must be in [0, 1]")

        def warmup_cosine_factor(epoch_index: int) -> float:
            completed_epoch = epoch_index + 1
            if warmup_epochs and completed_epoch <= warmup_epochs:
                return completed_epoch / warmup_epochs
            progress = (completed_epoch - warmup_epochs) / max(
                1, total_epochs - warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_factor)

    early_stopping_config = config["training"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_config.get("enabled", False))
    early_stopping_patience = int(early_stopping_config.get("patience", 7))
    early_stopping_min_delta = float(early_stopping_config.get("min_delta", 1e-4))
    epochs_without_improvement = 0

    checkpoint_metric = str(config["training"].get("checkpoint_metric", "loss_total"))
    checkpoint_mode = str(config["training"].get("checkpoint_mode", "min")).lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("training.checkpoint_mode must be 'min' or 'max'")

    start_epoch = 1
    best_val_loss = float("inf") if checkpoint_mode == "min" else -float("inf")

    init_checkpoint = config["training"].get("init_checkpoint")

    if init_checkpoint is not None and args.resume is None:
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        init_strict = bool(config["training"].get("init_strict", True))
        load_initial_model_state(model, checkpoint["model_state_dict"], init_strict)

        if ema_model is not None:
            load_initial_model_state(
                ema_model, checkpoint["model_state_dict"], init_strict
            )

        print(f"Initialised model weights from {init_checkpoint}")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if ema_model is not None and checkpoint.get("ema_model_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        if (
            quality_discriminator is not None
            and checkpoint.get("quality_discriminator_state_dict") is not None
        ):
            quality_discriminator.load_state_dict(checkpoint["quality_discriminator_state_dict"])
        if quality_optimizer is not None and checkpoint.get("quality_optimizer_state_dict") is not None:
            quality_optimizer.load_state_dict(checkpoint["quality_optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        restore_rng_state(checkpoint)
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
    print("MetricGAN-lite enabled:", metricgan_enabled)
    if metricgan_enabled:
        print("MetricGAN-lite weight:", metric_weight)
        print("MetricGAN-lite target:", quality_target_type)
        print("Quality update interval:", quality_update_interval)
        print(
            "Quality active epochs:",
            quality_start_epoch,
            "to",
            quality_stop_epoch if quality_stop_epoch is not None else "end",
        )
        print("Metric generator start epoch:", metric_generator_start_epoch)
        print("Quality discriminator parameters:", count_parameters(quality_discriminator))
    print("EMA enabled:", ema_enabled)
    print("Training precision:", configured_precision)
    print(f"Checkpoint selection: val_{checkpoint_metric} ({checkpoint_mode})")
    if ema_enabled:
        print("EMA decay:", ema_decay)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    epochs = int(args.epochs if args.epochs is not None else config["training"]["epochs"])
    perceptual_phase_scale = perceptual_config.get("phase_residual_scale")
    perceptual_magnitude_scale = perceptual_config.get("magnitude_residual_scale")
    if perceptual_enabled:
        print("Perceptual whole utterance:", perceptual_whole_utterance)
        print("Perceptual phase residual scale:", perceptual_phase_scale)
        print("Perceptual magnitude residual scale:", perceptual_magnitude_scale)
    scheduler_metric = str(lr_scheduler_config.get("metric", "loss_total"))
    subset_best = float("inf") if checkpoint_mode == "min" else -float("inf")
    full_candidates: list[tuple[float, int]] = []
    top3_manifest = checkpoint_dir / "full_validation_top3.yaml"
    if top3_manifest.exists():
        for candidate in yaml.safe_load(top3_manifest.read_text()) or []:
            full_candidates.append((float(candidate["score"]), int(candidate["epoch"])))
    safeguard_config = config["training"].get("checkpoint_safeguards", {})
    safeguard_baseline = None
    if safeguard_config:
        with Path(safeguard_config["baseline_summary"]).open() as file:
            raw_baseline = json.load(file)
        safeguard_baseline = raw_baseline.get("metrics", raw_baseline)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        metric_ramp_epochs = int(metricgan_config.get("generator_ramp_epochs", 0))
        if metric_ramp_epochs > 0:
            ramp_position = max(0, epoch - metric_generator_start_epoch + 1)
            epoch_metric_weight = metric_weight * min(1.0, ramp_position / metric_ramp_epochs)
        else:
            epoch_metric_weight = metric_weight
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            autocast_dtype=autocast_dtype,
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
            log_every=int(config["training"]["log_every"]),
            max_batches=max_train_batches,
            split_name="train",
            ema_model=ema_model,
            ema_decay=ema_decay,
            quality_discriminator=quality_discriminator,
            quality_optimizer=quality_optimizer,
            metric_weight=epoch_metric_weight,
            quality_grad_clip_norm=quality_grad_clip_norm,
            quality_target_type=quality_target_type,
            quality_update_interval=quality_update_interval,
            quality_start_epoch=quality_start_epoch,
            quality_stop_epoch=quality_stop_epoch,
            metric_generator_start_epoch=metric_generator_start_epoch,
            current_epoch=epoch,
            sample_rate=int(config["data"]["sample_rate"]),
            quality_target_floor=quality_target_floor,
            quality_target_ceiling=quality_target_ceiling,
            gradient_accumulation_steps=int(
                config["training"].get("gradient_accumulation_steps", 1)
            ),
        )

        should_validate = epoch % int(config["training"]["val_every"]) == 0
        eligible_checkpoint_candidate = True

        if should_validate:
            val_metrics = run_epoch(
                model=ema_model if ema_model is not None else model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                mixed_precision=False,
                autocast_dtype=None,
                grad_clip_norm=None,
                log_every=int(config["training"]["log_every"]),
                max_batches=max_val_batches,
                split_name="val",
            )
            if perceptual_enabled:
                perceptual_metrics = run_perceptual_validation(
                    model=ema_model if ema_model is not None else model,
                    loader=perceptual_val_loader,
                    device=device,
                    sample_rate=int(config["data"]["sample_rate"]),
                    max_items=int(perceptual_config.get("max_items", 50)),
                    phase_residual_scale=perceptual_phase_scale,
                    magnitude_residual_scale=perceptual_magnitude_scale,
                )
                val_metrics.update(perceptual_metrics)
            subset_value = val_metrics.get(checkpoint_metric)
            subset_improved = subset_value is not None and (
                subset_value < subset_best if checkpoint_mode == "min" else subset_value > subset_best
            )
            if subset_improved:
                subset_best = float(subset_value)
            if subset_improved and full_val_loader is not None:
                print("Subset leader: running full validation before checkpoint selection")
                full_metrics = run_epoch(
                    model=ema_model if ema_model is not None else model,
                    loader=full_val_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    scaler=None,
                    mixed_precision=False,
                    autocast_dtype=None,
                    grad_clip_norm=None,
                    log_every=int(config["training"]["log_every"]),
                    max_batches=None,
                    split_name="full_val",
                )
                if perceptual_enabled:
                    full_metrics.update(run_perceptual_validation(
                        model=ema_model if ema_model is not None else model,
                        loader=full_perceptual_val_loader,
                        device=device,
                        sample_rate=int(config["data"]["sample_rate"]),
                        max_items=len(full_val_loader.dataset),
                        phase_residual_scale=perceptual_phase_scale,
                        magnitude_residual_scale=perceptual_magnitude_scale,
                    ))
                val_metrics.update({f"full_{key}": value for key, value in full_metrics.items()})
                checkpoint_value = full_metrics.get(checkpoint_metric)
                if checkpoint_value is None:
                    raise KeyError(f"Full-validation metric missing: {checkpoint_metric}")
                val_metrics[checkpoint_metric] = checkpoint_value
                if safeguard_baseline is not None:
                    limits = {
                        "perceptual_enhanced_si_sdr": float(safeguard_config.get("max_si_sdr_drop", 0.15)),
                        "perceptual_enhanced_stoi": float(safeguard_config.get("max_stoi_drop", 0.002)),
                        "perceptual_enhanced_estoi": float(safeguard_config.get("max_estoi_drop", 0.003)),
                    }
                    violations = []
                    for metric_name, maximum_drop in limits.items():
                        baseline_name = metric_name.removeprefix("perceptual_")
                        if metric_name in full_metrics and baseline_name in safeguard_baseline:
                            drop = float(safeguard_baseline[baseline_name]) - float(full_metrics[metric_name])
                            if drop > maximum_drop:
                                violations.append(f"{baseline_name} drop {drop:.6f} > {maximum_drop:.6f}")
                    if violations:
                        eligible_checkpoint_candidate = False
                        print("Checkpoint rejected by perceptual safeguards:", "; ".join(violations))
            elif full_val_loader is not None:
                eligible_checkpoint_candidate = False
        else:
            val_metrics = {}

        val_loss = val_metrics.get("loss_total", float("inf"))
        checkpoint_value = val_metrics.get(checkpoint_metric)
        if val_metrics and not eligible_checkpoint_candidate:
            checkpoint_value = best_val_loss
        if val_metrics and checkpoint_value is None:
            raise KeyError(
                f"Checkpoint metric '{checkpoint_metric}' is not in validation metrics: "
                f"{sorted(val_metrics)}"
            )

        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }

        append_metrics_row(metrics_csv, row)

        print("Train:", train_metrics)

        if val_metrics:
            print("Val:", val_metrics)

        print(f"Learning rate: {current_lr:.2e}")

        improved = False
        if val_metrics:
            if scheduler is not None and lr_scheduler_config.get("name") == "reduce_on_plateau":
                if scheduler_metric not in val_metrics:
                    raise KeyError(
                        f"Scheduler metric '{scheduler_metric}' is not in validation metrics"
                    )
                scheduler.step(val_metrics[scheduler_metric])

            if checkpoint_mode == "min":
                improved = checkpoint_value < (best_val_loss - early_stopping_min_delta)
            else:
                improved = checkpoint_value > (best_val_loss + early_stopping_min_delta)

            if improved:
                best_val_loss = checkpoint_value
                epochs_without_improvement = 0

                print(
                    f"New best val_{checkpoint_metric}: {best_val_loss:.6f} "
                    f"({checkpoint_mode})"
                )
            else:
                epochs_without_improvement += 1

                if early_stopping_enabled:
                    print(
                        "No validation improvement: "
                        f"{epochs_without_improvement}/{early_stopping_patience}"
                    )

                    if epochs_without_improvement >= early_stopping_patience:
                        print("Early stopping triggered.")

        # Metric-independent schedules advance every epoch, including epochs
        # where validation is intentionally skipped.
        if scheduler is not None and lr_scheduler_config.get("name") != "reduce_on_plateau":
            scheduler.step()
        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "scheduler": scheduler,
            "scaler": scaler,
            "ema_model": ema_model,
            "epochs_without_improvement": epochs_without_improvement,
            "quality_discriminator": quality_discriminator,
            "quality_optimizer": quality_optimizer,
        }
        save_checkpoint(path=checkpoint_dir / "latest.pt", **checkpoint_arguments)
        if bool(config["training"].get("save_every_epoch", False)):
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:03d}.pt",
                **checkpoint_arguments,
            )
        if full_val_loader is not None and val_metrics and eligible_checkpoint_candidate:
            candidate_path = checkpoint_dir / f"full_candidate_epoch_{epoch:03d}.pt"
            save_checkpoint(path=candidate_path, **checkpoint_arguments)
            full_candidates.append((float(checkpoint_value), epoch))
            full_candidates.sort(key=lambda item: item[0], reverse=checkpoint_mode == "max")
            full_candidates = full_candidates[:3]
            keep_epochs = {candidate_epoch for _, candidate_epoch in full_candidates}
            for stale_path in checkpoint_dir.glob("full_candidate_epoch_*.pt"):
                stale_epoch = int(stale_path.stem.rsplit("_", 1)[-1])
                if stale_epoch not in keep_epochs:
                    stale_path.unlink()
            (checkpoint_dir / "full_validation_top3.yaml").write_text(
                yaml.safe_dump([
                    {"rank": rank, "epoch": candidate_epoch, "score": score,
                     "checkpoint": f"full_candidate_epoch_{candidate_epoch:03d}.pt"}
                    for rank, (score, candidate_epoch) in enumerate(full_candidates, start=1)
                ], sort_keys=False)
            )
        if improved:
            save_checkpoint(path=checkpoint_dir / "best.pt", **checkpoint_arguments)
        if (
            val_metrics
            and early_stopping_enabled
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
