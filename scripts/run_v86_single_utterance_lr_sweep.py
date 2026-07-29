#!/usr/bin/env python3
"""FP32 one-utterance mask-overfit sweep before any larger V8 continuation."""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v8/generated/v85_ratio_capacity/direct_ratio.yaml"
RESULT_ROOT = ROOT / "results/v86/single_utterance_lr_sweep"
CHECKPOINT_ROOT = ROOT / "checkpoints/v86"
NEXT_CONFIG_ROOT = ROOT / "configs/v8/generated/v86_promoted"
LEARNING_RATES = (1e-5, 3e-5, 1e-4)
EVALUATION_STEPS = (0, 25, 50, 100, 200, 300, 400, 500)
MAX_UPDATES = 500
SEED = 8601


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ratio_objective(
    predicted_mask: torch.Tensor,
    target_mask: torch.Tensor,
    importance: torch.Tensor,
) -> torch.Tensor:
    return (
        F.smooth_l1_loss(predicted_mask.float(), target_mask, reduction="none") * importance
    ).mean()


def speech_metrics(noisy: torch.Tensor, clean: torch.Tensor, enhanced: torch.Tensor) -> dict[str, float]:
    return compute_speech_metrics(
        noisy=noisy.squeeze().detach().float().cpu().numpy(),
        enhanced=enhanced.squeeze().detach().float().cpu().numpy(),
        clean=clean.squeeze().detach().float().cpu().numpy(),
        sample_rate=16000,
    )


def build_targets(model, noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        noisy_spectrum, _ = model._analysis(noisy.squeeze(1), pad_end=True)
        clean_spectrum, _ = model._analysis(clean.squeeze(1), pad_end=True)
        noisy_magnitude = noisy_spectrum.abs().clamp_min(1e-7)
        clean_magnitude = clean_spectrum.abs().clamp_min(1e-7)
        target = (clean_magnitude / noisy_magnitude).clamp(0.0, 1.0)
        importance = clean_magnitude.pow(0.3)
        importance = importance / importance.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        importance = importance.clamp(0.1, 5.0)
    return target, importance


def evaluate(model, noisy, clean, target, importance, step: int) -> dict[str, float | int]:
    model.eval()
    with torch.no_grad():
        output = model(noisy)
        loss = ratio_objective(output.magnitude_mask, target, importance)
    metrics = speech_metrics(noisy, clean, output.enhanced)
    return {
        "step": step,
        "ratio_loss": float(loss),
        "pesq": float(metrics["enhanced_pesq"]),
        "stoi": float(metrics["enhanced_stoi"]),
        "estoi": float(metrics["enhanced_estoi"]),
        "si_sdr": float(metrics["enhanced_si_sdr"]),
    }


def run_candidate(
    model_config: dict,
    initial_state: dict[str, torch.Tensor],
    noisy: torch.Tensor,
    clean: torch.Tensor,
    learning_rate: float,
) -> dict[str, object]:
    name = f"lr_{learning_rate:.0e}".replace("-", "m")
    model = build_model(model_config).to(noisy.device)
    model.load_state_dict(initial_state)
    target, importance = build_targets(model, noisy, clean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history = [evaluate(model, noisy, clean, target, importance, 0)]
    best = history[0]
    finite = True
    maximum_preclip_gradient_norm = 0.0
    checkpoint_dir = CHECKPOINT_ROOT / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": 0, "config": {"model": model_config}, "model_state_dict": model.state_dict()},
        checkpoint_dir / "epoch_000.pt",
    )

    for step in range(1, MAX_UPDATES + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(noisy)
        loss = ratio_objective(output.magnitude_mask, target, importance)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not gradients_finite:
            finite = False
            break
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        maximum_preclip_gradient_norm = max(maximum_preclip_gradient_norm, float(norm))
        optimizer.step()

        if step in EVALUATION_STEPS:
            current = evaluate(model, noisy, clean, target, importance, step)
            history.append(current)
            if float(current["ratio_loss"]) < float(best["ratio_loss"]):
                best = current
                torch.save(
                    {
                        "epoch": step,
                        "config": {"model": model_config},
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_dir / "best.pt",
                )

    if not (checkpoint_dir / "best.pt").exists():
        torch.save(
            {"epoch": 0, "config": {"model": model_config}, "model_state_dict": initial_state},
            checkpoint_dir / "best.pt",
        )
    improvements = [
        float(history[index]["ratio_loss"]) <= float(history[index - 1]["ratio_loss"])
        for index in range(1, len(history))
    ]
    monotonic_fraction = sum(improvements) / max(1, len(improvements))
    passed = bool(
        finite
        and float(best["ratio_loss"]) < 0.02
        and float(best["pesq"]) > float(history[0]["pesq"])
        and monotonic_fraction >= 0.6
    )
    result = {
        "name": name,
        "learning_rate": learning_rate,
        "finite": finite,
        "maximum_preclip_gradient_norm": maximum_preclip_gradient_norm,
        "monotonic_fraction": monotonic_fraction,
        "initial": history[0],
        "best": best,
        "history": history,
        "passed": passed,
    }
    candidate_dir = RESULT_ROOT / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_promoted_configs(base: dict, winner: dict[str, object]) -> list[str]:
    paths = []
    for precision in ("bf16", "fp32"):
        config = copy.deepcopy(base)
        config["project"]["seed"] = 8611
        config["model"]["scale_preserving_detail"] = False
        config["training"]["learning_rate"] = float(winner["learning_rate"])
        config["training"]["precision"] = precision
        config["training"]["epochs"] = 40
        config["training"]["val_every"] = 1
        config["training"]["grad_clip_norm"] = 1.0
        config["training"]["checkpoint_metric"] = "loss_magnitude_ratio"
        config["training"]["checkpoint_mode"] = "min"
        config["training"]["early_stopping"] = {
            "enabled": True, "patience": 8, "min_delta": 1e-4
        }
        config["training"]["lr_scheduler"] = {"name": "none"}
        name = f"ratio_16_{precision}"
        config["paths"]["checkpoint_dir"] = "checkpoints/v86"
        config["paths"]["experiment_name"] = name
        path = NEXT_CONFIG_ROOT / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(str(path.relative_to(ROOT)))
    return paths


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This sweep requires CUDA")
    base = yaml.safe_load(BASE_CONFIG.read_text())
    model_config = copy.deepcopy(base["model"])
    model_config["scale_preserving_detail"] = False
    seed_all(SEED)
    template = build_model(model_config).cuda()
    initial_state = copy.deepcopy(template.state_dict())
    del template

    dataset = PairedSpeechDataset(
        base["data"]["train_metadata"],
        sample_rate=16000,
        chunk_seconds=4.0,
        random_crop=False,
    )
    item = dataset[0]
    noisy = item["noisy"].unsqueeze(0).cuda()
    clean = item["clean"].unsqueeze(0).cuda()
    results = []
    for learning_rate in LEARNING_RATES:
        seed_all(SEED)
        result = run_candidate(model_config, initial_state, noisy, clean, learning_rate)
        results.append(result)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        (RESULT_ROOT / "status.json").write_text(
            json.dumps({"complete": False, "results": results}, indent=2) + "\n"
        )

    passing = [result for result in results if bool(result["passed"])]
    winner = min(passing, key=lambda row: float(row["best"]["ratio_loss"])) if passing else None
    promoted_configs = write_promoted_configs(base, winner) if winner is not None else []
    status = {
        "complete": True,
        "updates_per_candidate": MAX_UPDATES,
        "file_id": item["file_id"],
        "results": results,
        "winner": winner,
        "micro_overfit_gate": winner is not None,
        "promoted_configs": promoted_configs,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
