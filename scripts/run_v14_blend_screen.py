#!/usr/bin/env python3
"""Screen conservative output blending for PESQ gain under no-harm gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TEST_SHA256 = (
    "33454f6151a30bbae211a74c54c118f48413ee02a396943d0ed8754c08bff679"
)
METRICS = (
    "enhanced_pesq",
    "enhanced_si_sdr",
    "enhanced_stoi",
    "enhanced_estoi",
)
NOISY_METRICS = {
    "enhanced_pesq": "noisy_pesq",
    "enhanced_si_sdr": "noisy_si_sdr",
    "enhanced_stoi": "noisy_stoi",
    "enhanced_estoi": "noisy_estoi",
}
SAFEGUARDS = {
    "enhanced_si_sdr": 0.10,
    "enhanced_stoi": 0.001,
    "enhanced_estoi": 0.002,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(strength: float) -> str:
    return f"strength_{round(100 * strength):03d}"


def _tensor_to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return (
        waveform.squeeze().detach().cpu().numpy().astype(np.float32)
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    excluded = {"file_id", "speaker_id"}
    result: dict[str, float | None] = {}
    for metric in rows[0]:
        if metric in excluded:
            continue
        values = [
            float(row[metric])
            for row in rows
            if row[metric] is not None
        ]
        result[metric] = float(np.mean(values)) if values else None
    return result


def _paired_report(
    rows_by_strength: dict[float, list[dict[str, Any]]],
    *,
    strengths: list[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
    minimum_pesq_gain: float,
    max_harm_rate_increase: float,
) -> dict[str, Any]:
    reference_strength = 1.0
    reference_rows = rows_by_strength[reference_strength]
    reference = {
        metric: np.asarray([row[metric] for row in reference_rows], dtype=float)
        for metric in METRICS
    }
    noisy = {
        metric: np.asarray(
            [row[NOISY_METRICS[metric]] for row in reference_rows],
            dtype=float,
        )
        for metric in METRICS
    }
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(reference_rows),
        size=(bootstrap_samples, len(reference_rows)),
    )
    report: dict[str, Any] = {
        "reference_strength": reference_strength,
        "minimum_pesq_gain": minimum_pesq_gain,
        "safeguards": {
            "maximum_metric_drop": SAFEGUARDS,
            "maximum_harm_rate_increase": max_harm_rate_increase,
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "candidates": {},
        "winner": None,
    }
    winners: list[tuple[float, float]] = []
    for strength in strengths:
        if strength == reference_strength:
            continue
        candidate_rows = rows_by_strength[strength]
        candidate = {
            metric: np.asarray(
                [row[metric] for row in candidate_rows], dtype=float
            )
            for metric in METRICS
        }
        metric_report: dict[str, Any] = {}
        safeguard_passes = True
        harm_passes = True
        for metric in METRICS:
            delta = candidate[metric] - reference[metric]
            bootstrap = delta[indices].mean(axis=1)
            harm_rate = float(np.mean(candidate[metric] < noisy[metric]))
            reference_harm_rate = float(
                np.mean(reference[metric] < noisy[metric])
            )
            metric_report[metric] = {
                "reference_mean": float(reference[metric].mean()),
                "candidate_mean": float(candidate[metric].mean()),
                "delta_mean": float(delta.mean()),
                "paired_bootstrap_ci95": np.quantile(
                    bootstrap, [0.025, 0.975]
                ).tolist(),
                "reference_harm_rate": reference_harm_rate,
                "candidate_harm_rate": harm_rate,
                "harm_rate_delta": harm_rate - reference_harm_rate,
            }
            if metric in SAFEGUARDS:
                safeguard_passes &= (
                    float(delta.mean()) >= -SAFEGUARDS[metric]
                )
            harm_passes &= (
                harm_rate - reference_harm_rate
                <= max_harm_rate_increase
            )
        pesq = metric_report["enhanced_pesq"]
        pesq_passes = (
            pesq["delta_mean"] >= minimum_pesq_gain
            and pesq["paired_bootstrap_ci95"][0] > 0.0
        )
        promoted = bool(pesq_passes and safeguard_passes and harm_passes)
        report["candidates"][_label(strength)] = {
            "strength": strength,
            "metrics": metric_report,
            "pesq_gate_passes": bool(pesq_passes),
            "metric_safeguards_pass": bool(safeguard_passes),
            "harm_rate_safeguards_pass": bool(harm_passes),
            "promoted": promoted,
        }
        if promoted:
            winners.append((float(pesq["delta_mean"]), strength))
    if winners:
        report["winner"] = _label(max(winners)[1])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v12/full/gru_matched_time1_seed1200/best.pt"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/processed/voicebank_demand/metadata/"
            "v12_architecture_selection_400.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v14/blend_screen_seed1200"),
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.80, 0.90, 0.95, 1.00],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=14_000)
    parser.add_argument("--minimum-pesq-gain", type=float, default=0.01)
    parser.add_argument(
        "--max-harm-rate-increase", type=float, default=0.01
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, validate inputs and print the frozen screen.",
    )
    args = parser.parse_args()
    strengths = sorted(set(float(value) for value in args.strengths))
    if 1.0 not in strengths:
        parser.error("--strengths must include the 1.0 reference")
    if any(value <= 0.0 or value > 1.0 for value in strengths):
        parser.error("strengths must be in (0, 1]")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    checkpoint_path = ROOT / args.checkpoint
    metadata_path = ROOT / args.metadata
    for path in (checkpoint_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(metadata_path) == OFFICIAL_TEST_SHA256:
        raise ValueError(
            "The standard test set is forbidden for V14 screening"
        )
    plan = {
        "purpose": "diagnostic conservative-blend screen",
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "metadata_sha256": _sha256(metadata_path),
        "standard_test_used": False,
        "strengths": strengths,
        "formula": "blended = noisy + strength * (enhanced - noisy)",
        "minimum_pesq_gain": args.minimum_pesq_gain,
        "maximum_metric_drop": SAFEGUARDS,
        "maximum_harm_rate_increase": args.max_harm_rate_increase,
        "device": args.device,
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return

    device = torch.device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = PairedSpeechDataset(
        metadata_csv=metadata_path,
        project_root=ROOT,
        sample_rate=16000,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    rows_by_strength: dict[float, list[dict[str, Any]]] = {
        strength: [] for strength in strengths
    }
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = _tensor_to_numpy(item["clean"])
            noisy_np = _tensor_to_numpy(noisy)
            enhanced = model(noisy).enhanced
            for strength in strengths:
                blended = noisy + strength * (enhanced - noisy)
                metrics = compute_speech_metrics(
                    noisy=noisy_np,
                    enhanced=_tensor_to_numpy(blended),
                    clean=clean,
                    sample_rate=16000,
                )
                rows_by_strength[strength].append(
                    {
                        "file_id": item["file_id"],
                        "speaker_id": item["speaker_id"],
                        **metrics,
                    }
                )

    output_dir = ROOT / args.output_dir
    for strength, rows in rows_by_strength.items():
        directory = output_dir / _label(strength)
        _write_csv(directory / "per_file_metrics.csv", rows)
        summary = {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "num_items": len(rows),
            "strength": strength,
            "formula": plan["formula"],
            "metrics": _mean_metrics(rows),
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    decision = _paired_report(
        rows_by_strength,
        strengths=strengths,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        minimum_pesq_gain=args.minimum_pesq_gain,
        max_harm_rate_increase=args.max_harm_rate_increase,
    )
    decision["plan"] = plan
    (output_dir / "decision_report.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
