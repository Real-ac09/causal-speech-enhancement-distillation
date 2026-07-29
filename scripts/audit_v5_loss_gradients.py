#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from cnvqg.data import PairedSpeechDataset, speech_enhancement_collate_fn
from cnvqg.losses import EnhancementLoss
from cnvqg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V5 objective gradient cosine similarity.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--num-items", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device).train()
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = EnhancementLoss(**config["loss"]).to(device)
    dataset = PairedSpeechDataset(
        config["data"]["train_metadata"], chunk_seconds=4.0, random_crop=False
    )
    names = ["waveform_l1", "si_sdr", "stft", "mel", "complex_stft", "noise_prediction",
             "magnitude", "phase", "group_delay", "instantaneous_frequency",
             "phase_confidence", "vq"]
    weights = {
        "waveform_l1": criterion.waveform_l1_weight,
        "si_sdr": criterion.si_sdr_weight,
        "stft": criterion.stft_weight,
        "mel": criterion.mel_weight,
        "complex_stft": criterion.complex_stft_weight,
        "noise_prediction": criterion.noise_prediction_weight,
        "magnitude": criterion.magnitude_weight,
        "phase": criterion.phase_weight,
        "group_delay": criterion.group_delay_weight,
        "instantaneous_frequency": criterion.instantaneous_frequency_weight,
        "phase_confidence": criterion.phase_confidence_weight,
        "vq": criterion.vq_weight,
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    cosine_samples, norm_samples = {}, {name: [] for name in names}
    count = min(args.num_items, len(dataset))
    for item_index in range(count):
        batch = speech_enhancement_collate_fn([dataset[item_index]])
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        output = model(noisy)
        losses = criterion(
            output.enhanced.float(), clean.float(), output.vq.loss.float(), noisy=noisy.float(),
            noise_prediction=output.noise_prediction.float(),
            estimated_magnitude=output.estimated_magnitude.float(),
            estimated_phase=output.estimated_phase.float(),
            phase_confidence=(
                output.phase_confidence.float()
                if getattr(output, "phase_confidence", None) is not None
                else None
            ),
            phase_candidate=output.phase_candidate.float(),
        )
        gradients = {}
        for name in names:
            if weights[name] <= 0.0:
                continue
            value = getattr(losses, name)
            if not value.requires_grad or float(value.detach()) == 0.0:
                continue
            parts = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            gradient = torch.cat([
                (part if part is not None else torch.zeros_like(parameter)).reshape(-1)
                for part, parameter in zip(parts, parameters)
            ])
            gradients[name] = gradient
            norm_samples[name].append(float(gradient.norm().detach().cpu()))
        for index, left in enumerate(gradients):
            for right in list(gradients)[index + 1:]:
                key = f"{left}:{right}"
                cosine_samples.setdefault(key, []).append(
                    torch.nn.functional.cosine_similarity(
                        gradients[left], gradients[right], dim=0, eps=1e-12
                    ).item()
                )
        del output, losses, gradients

    cosine = {key: sum(values) / len(values) for key, values in cosine_samples.items()}
    gradient_norm = {
        name: sum(values) / len(values) for name, values in norm_samples.items() if values
    }
    weighted_gradient_norm = {
        name: norm * weights[name] for name, norm in gradient_norm.items()
    }
    redundant = [
        {"pair": key.split(":"), "cosine": score}
        for key, score in cosine.items() if score >= args.threshold
    ]
    report = {"threshold": args.threshold, "num_items": count, "cosine": cosine,
              "gradient_norm": gradient_norm,
              "configured_weight": weights,
              "weighted_gradient_norm": weighted_gradient_norm,
              "near_duplicate_objectives": redundant,
              "action": "Remove only after confirming across multiple batches and seeds."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
