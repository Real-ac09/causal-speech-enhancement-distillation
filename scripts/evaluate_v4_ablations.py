#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models.factory import build_model


ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {
        "use_noise_codebook": True,
        "condition_dynamics": True,
        "adaptive_iterations": True,
        "max_iterations": 4,
    },
    "no_vq": {
        "use_noise_codebook": False,
        "condition_dynamics": True,
        "adaptive_iterations": True,
        "max_iterations": 4,
    },
    "no_condition": {
        "use_noise_codebook": True,
        "condition_dynamics": False,
        "adaptive_iterations": True,
        "max_iterations": 4,
    },
    "fixed_1": {
        "use_noise_codebook": True,
        "condition_dynamics": True,
        "adaptive_iterations": False,
        "max_iterations": 1,
    },
    "fixed_2": {
        "use_noise_codebook": True,
        "condition_dynamics": True,
        "adaptive_iterations": False,
        "max_iterations": 2,
    },
    "fixed_4": {
        "use_noise_codebook": True,
        "condition_dynamics": True,
        "adaptive_iterations": False,
        "max_iterations": 4,
    },
    "continuous_adaptive_2": {
        "use_noise_codebook": False,
        "condition_dynamics": True,
        "adaptive_iterations": True,
        "max_iterations": 2,
    },
    "continuous_adaptive_3": {
        "use_noise_codebook": False,
        "condition_dynamics": True,
        "adaptive_iterations": True,
        "max_iterations": 3,
    },
    "continuous_adaptive_4": {
        "use_noise_codebook": False,
        "condition_dynamics": True,
        "adaptive_iterations": True,
        "max_iterations": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4 inference ablations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(ABLATIONS),
        default=None,
    )
    mode_group.add_argument(
        "--gate-values",
        nargs="+",
        default=None,
        metavar="GATE",
        help=(
            "Run a gate-only V4.3 ablation while preserving all other checkpoint "
            "settings. Values are 'learned', 'off', or numbers in [0, 1]."
        ),
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_model(model: torch.nn.Module, mode: str) -> None:
    settings = ABLATIONS[mode]
    model.use_noise_codebook = settings["use_noise_codebook"]
    model.condition_dynamics = settings["condition_dynamics"]
    model.cell.condition_dynamics = settings["condition_dynamics"]
    model.adaptive_iterations = settings["adaptive_iterations"]
    model.max_iterations = settings["max_iterations"]


def parse_gate_modes(values: list[str]) -> list[tuple[str, float | None]]:
    modes: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for raw_value in values:
        value = raw_value.lower()
        if value == "learned":
            name, gate = "learned", None
        elif value in {"off", "none", "disabled"}:
            name, gate = "off", 0.0
        else:
            try:
                gate = float(value)
            except ValueError as error:
                raise ValueError(f"Invalid gate value: {raw_value!r}") from error
            if not 0.0 <= gate <= 1.0:
                raise ValueError(f"Gate value must be in [0, 1], got {gate}")
            name = f"gate_{gate:g}".replace(".", "p")
        if name in seen:
            raise ValueError(f"Duplicate gate mode: {name}")
        seen.add(name)
        modes.append((name, gate))
    return modes


def configure_gate(model: torch.nn.Module, gate: float | None) -> dict[str, Any]:
    if not hasattr(model, "vq_gate_logits"):
        raise TypeError("Gate ablations require a model with vq_gate_logits.")
    if gate is None:
        model.use_noise_codebook = True
        learned_gate = torch.sigmoid(model.vq_gate_logits.detach().float())
        return {
            "gate": "learned",
            "gate_mean": float(learned_gate.mean()),
            "gate_min": float(learned_gate.min()),
            "gate_max": float(learned_gate.max()),
            "use_noise_codebook": True,
        }
    if gate == 0.0:
        model.use_noise_codebook = False
        return {"gate": 0.0, "use_noise_codebook": False}
    if model.vq_gate_logits is None:
        raise TypeError("Forced gate ablations require a learnable VQ gate checkpoint.")
    model.use_noise_codebook = True
    logit = math.inf if gate == 1.0 else math.log(gate / (1.0 - gate))
    with torch.no_grad():
        model.vq_gate_logits.fill_(logit)
    return {"gate": gate, "use_noise_codebook": True}


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not hasattr(model, "cell") or not hasattr(model, "max_iterations"):
        raise TypeError("This evaluator requires a NoiseAdaptiveTFMamba checkpoint.")

    dataset = PairedSpeechDataset(
        metadata_csv=args.metadata,
        project_root=".",
        sample_rate=args.sample_rate,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    num_items = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}

    if args.gate_values is not None:
        evaluation_modes = parse_gate_modes(args.gate_values)
    else:
        evaluation_modes = [
            (mode, None) for mode in (args.modes if args.modes is not None else ABLATIONS)
        ]

    for mode, gate in evaluation_modes:
        # Restore the exact checkpoint before every mode so a forced gate cannot
        # leak into the learned-gate result or a later structural ablation.
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.gate_values is not None:
            settings = configure_gate(model, gate)
        else:
            configure_model(model, mode)
            settings = ABLATIONS[mode]
        rows: list[dict[str, Any]] = []
        expected_iterations: list[float] = []
        halting_probabilities: list[torch.Tensor] = []
        code_counts: collections.Counter[int] = collections.Counter()
        inference_seconds = 0.0
        audio_seconds = 0.0

        with torch.inference_mode():
            # Exclude lazy CUDA/Mamba kernel initialisation from the timing.
            warmup_noisy = dataset[0]["noisy"].unsqueeze(0).to(device)
            for _ in range(3):
                model(warmup_noisy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for index in tqdm(range(num_items), desc=mode):
                item = dataset[index]
                noisy = item["noisy"].unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                output = model(noisy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - start_time
                audio_seconds += float(noisy.shape[-1]) / args.sample_rate
                enhanced = output.enhanced.squeeze().float().cpu().numpy()
                noisy_np = item["noisy"].squeeze().numpy().astype(np.float32)
                clean_np = item["clean"].squeeze().numpy().astype(np.float32)
                metrics = compute_speech_metrics(
                    noisy=noisy_np,
                    enhanced=enhanced.astype(np.float32),
                    clean=clean_np,
                    sample_rate=args.sample_rate,
                )
                rows.append(
                    {
                        "file_id": item["file_id"],
                        "speaker_id": item["speaker_id"],
                        **metrics,
                    }
                )
                expected_iterations.append(float(output.expected_iterations.mean()))
                halting_probabilities.append(
                    output.halting_probabilities.squeeze(0).float().cpu()
                )
                valid_codes = output.code_indices.flatten().cpu().tolist()
                code_counts.update(int(code) for code in valid_codes if code >= 0)

        metric_names = [
            name for name in rows[0] if name not in {"file_id", "speaker_id"}
        ]
        total_codes = sum(code_counts.values())
        if total_codes:
            probabilities = torch.tensor(list(code_counts.values()), dtype=torch.float32)
            probabilities /= probabilities.sum()
            code_perplexity = float(
                torch.exp(-(probabilities * (probabilities + 1e-10).log()).sum())
            )
        else:
            code_perplexity = 0.0
        max_depth = max(probability.numel() for probability in halting_probabilities)
        padded_halting = [
            torch.nn.functional.pad(probability, (0, max_depth - probability.numel()))
            for probability in halting_probabilities
        ]
        summary = {
            "settings": settings,
            "num_items": num_items,
            "metrics": {name: mean([float(row[name]) for row in rows]) for name in metric_names},
            "expected_iterations": mean(expected_iterations),
            "mean_halting_probabilities": torch.stack(padded_halting).mean(0).tolist(),
            "inference_seconds": inference_seconds,
            "audio_seconds": audio_seconds,
            "real_time_factor": inference_seconds / audio_seconds,
            "unique_codes": len(code_counts),
            "global_code_perplexity": code_perplexity,
            "top_codes": code_counts.most_common(10),
        }
        summaries[mode] = summary
        write_csv(args.output_dir / mode / "per_file_metrics.csv", rows)
        with (args.output_dir / mode / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2)
        print(mode, json.dumps(summary, indent=2))

    with (args.output_dir / "comparison.json").open("w") as file:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "metadata": str(args.metadata),
                "ablations": summaries,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
