#!/usr/bin/env python3
"""Export a fixed V13 listening set with mask and spectrogram diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cnvqg.data import PairedSpeechDataset
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    Path("checkpoints/v12/full/gru_matched_time1_seed1200/best.pt"),
    Path("checkpoints/v12/full/gru_matched_time1_seed1201/best.pt"),
    Path("checkpoints/v12/full/gru_matched_time1_seed1202/best.pt"),
)


def _select_cases(frame: pd.DataFrame) -> pd.DataFrame:
    selections: dict[str, set[str]] = defaultdict(set)

    def add(category: str, selected: pd.DataFrame) -> None:
        for file_id in selected["file_id"]:
            selections[str(file_id)].add(category)

    add("pesq_harm", frame.nsmallest(5, "pesq_gain_mean"))
    intelligibility = frame.assign(
        intelligibility_harm_score=frame["stoi_gain_mean"]
        + frame["estoi_gain_mean"]
    )
    add(
        "intelligibility_harm",
        intelligibility.nsmallest(5, "intelligibility_harm_score"),
    )
    add(
        "pesq_down_intelligibility_up",
        frame[
            (frame["pesq_gain_mean"] < 0.0)
            & (frame["stoi_gain_mean"] > 0.0)
            & (frame["estoi_gain_mean"] > 0.0)
        ].nsmallest(3, "pesq_gain_mean"),
    )
    add(
        "pesq_up_intelligibility_down",
        intelligibility[
            (intelligibility["pesq_gain_mean"] > 0.0)
            & (intelligibility["stoi_gain_mean"] < 0.0)
            & (intelligibility["estoi_gain_mean"] < 0.0)
        ].nsmallest(3, "intelligibility_harm_score"),
    )

    variability_columns = [
        "enhanced_pesq_seed_std",
        "enhanced_si_sdr_seed_std",
        "enhanced_stoi_seed_std",
        "enhanced_estoi_seed_std",
    ]
    variability_rank = sum(
        frame[column].rank(pct=True) for column in variability_columns
    )
    add(
        "seed_sensitive",
        frame.assign(variability_rank=variability_rank).nlargest(
            4, "variability_rank"
        ),
    )

    gain_columns = [
        "pesq_gain_mean",
        "si_sdr_gain_mean",
        "stoi_gain_mean",
        "estoi_gain_mean",
    ]
    median_distance = np.zeros(len(frame), dtype=np.float64)
    for column in gain_columns:
        scale = frame[column].std()
        median_distance += np.square(
            (frame[column] - frame[column].median()) / max(scale, 1e-12)
        )
    add(
        "typical",
        frame.assign(median_distance=median_distance).nsmallest(
            4, "median_distance"
        ),
    )
    add("best_pesq_gain", frame.nlargest(4, "pesq_gain_mean"))

    order = {
        "pesq_harm": 0,
        "intelligibility_harm": 1,
        "pesq_down_intelligibility_up": 2,
        "pesq_up_intelligibility_down": 3,
        "seed_sensitive": 4,
        "typical": 5,
        "best_pesq_gain": 6,
    }
    selected = frame[frame["file_id"].isin(selections)].copy()
    selected["categories"] = selected["file_id"].map(
        lambda file_id: ";".join(
            sorted(selections[str(file_id)], key=order.__getitem__)
        )
    )
    selected["category_order"] = selected["file_id"].map(
        lambda file_id: min(order[value] for value in selections[str(file_id)])
    )
    selected = selected.sort_values(
        ["category_order", "pesq_gain_mean", "file_id"]
    ).reset_index(drop=True)
    selected.insert(0, "case_id", [f"CASE{index:03d}" for index in range(1, len(selected) + 1)])
    return selected.drop(columns="category_order")


def _save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def _db(magnitude: np.ndarray, reference: float) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(magnitude, reference * 1e-4) / reference)


def _plot_diagnostic(
    path: Path,
    *,
    noisy_spectrum: torch.Tensor,
    clean_spectrum: torch.Tensor,
    enhanced_spectrum: torch.Tensor,
    mask: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    file_id: str,
    categories: str,
) -> None:
    noisy = noisy_spectrum.squeeze(0).abs().detach().cpu().numpy()
    clean = clean_spectrum.squeeze(0).abs().detach().cpu().numpy()
    enhanced = enhanced_spectrum.squeeze(0).abs().detach().cpu().numpy()
    mask_values = mask.squeeze(0).detach().cpu().numpy()
    bins = min(noisy.shape[0], clean.shape[0], enhanced.shape[0], mask_values.shape[0])
    frames = min(noisy.shape[1], clean.shape[1], enhanced.shape[1], mask_values.shape[1])
    noisy, clean, enhanced, mask_values = (
        value[:bins, :frames] for value in (noisy, clean, enhanced, mask_values)
    )
    reference = max(float(noisy.max()), float(clean.max()), float(enhanced.max()), 1e-7)
    extent = [
        0.0,
        frames * hop_length / sample_rate,
        0.0,
        sample_rate / 2.0,
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13, 7.5), layout="constrained")
    for axis, title, value in (
        (axes[0, 0], "Noisy magnitude", noisy),
        (axes[0, 1], "Enhanced magnitude (seed 1200)", enhanced),
        (axes[1, 0], "Clean magnitude", clean),
    ):
        image = axis.imshow(
            _db(value, reference),
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=-80.0,
            vmax=0.0,
            cmap="magma",
        )
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")
    magnitude_divider = make_axes_locatable(axes[0, 1])
    magnitude_axis = magnitude_divider.append_axes("right", size="3%", pad=0.06)
    magnitude_colourbar = figure.colorbar(image, cax=magnitude_axis)
    magnitude_colourbar.set_label("Magnitude (dB, common reference)")
    mask_image = axes[1, 1].imshow(
        mask_values,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=2.0,
        cmap="coolwarm",
    )
    axes[1, 1].set_title("Predicted scalar magnitude mask")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frequency (Hz)")
    mask_divider = make_axes_locatable(axes[1, 1])
    mask_axis = mask_divider.append_axes("right", size="3%", pad=0.06)
    mask_colourbar = figure.colorbar(mask_image, cax=mask_axis)
    mask_colourbar.set_label("Mask value")
    figure.suptitle(f"{file_id}: {categories}")
    figure.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def _mask_diagnostics(
    *,
    noisy_spectrum: torch.Tensor,
    clean_spectrum: torch.Tensor,
    enhanced_spectrum: torch.Tensor,
    mask: torch.Tensor,
    sample_rate: int,
) -> dict[str, float]:
    noisy = noisy_spectrum.abs().float()
    clean = clean_spectrum.abs().float()
    enhanced = enhanced_spectrum.abs().float()
    predicted_mask = mask.float()
    bins = min(noisy.shape[-2], clean.shape[-2], enhanced.shape[-2], predicted_mask.shape[-2])
    frames = min(noisy.shape[-1], clean.shape[-1], enhanced.shape[-1], predicted_mask.shape[-1])
    noisy = noisy[..., :bins, :frames]
    clean = clean[..., :bins, :frames]
    enhanced = enhanced[..., :bins, :frames]
    predicted_mask = predicted_mask[..., :bins, :frames]
    target_mask = (clean / noisy.clamp_min(1e-7)).clamp(0.0, 1.0)
    clean_weight = clean.pow(0.3)
    clean_weight = clean_weight / clean_weight.sum().clamp_min(1e-7)
    frequencies = torch.linspace(
        0.0,
        sample_rate / 2.0,
        bins,
        device=predicted_mask.device,
    )
    clean_dominated = target_mask >= 0.8
    noise_dominated = target_mask <= 0.3
    log_noisy_error = (
        torch.log1p(noisy) - torch.log1p(clean)
    ).abs().mean()
    log_enhanced_error = (
        torch.log1p(enhanced) - torch.log1p(clean)
    ).abs().mean()
    return {
        "mask_mean": float(predicted_mask.mean()),
        "mask_p10": float(torch.quantile(predicted_mask, 0.10)),
        "mask_p50": float(torch.quantile(predicted_mask, 0.50)),
        "mask_p90": float(torch.quantile(predicted_mask, 0.90)),
        "mask_fraction_below_0_5": float((predicted_mask < 0.5).float().mean()),
        "mask_fraction_below_0_8": float((predicted_mask < 0.8).float().mean()),
        "mask_fraction_above_1_0": float((predicted_mask > 1.0).float().mean()),
        "clean_weighted_mask_mean": float((predicted_mask * clean_weight).sum()),
        "low_band_mask_mean": float(predicted_mask[..., frequencies < 500.0, :].mean()),
        "high_band_mask_mean": float(predicted_mask[..., frequencies >= 4000.0, :].mean()),
        "clean_dominated_oversuppression_rate": float(
            ((predicted_mask < 0.7) & clean_dominated).float().sum()
            / clean_dominated.float().sum().clamp_min(1.0)
        ),
        "noise_dominated_undersuppression_rate": float(
            ((predicted_mask > 0.7) & noise_dominated).float().sum()
            / noise_dominated.float().sum().clamp_min(1.0)
        ),
        "mask_mae_to_capped_oracle": float(
            ((predicted_mask - target_mask).abs() * clean_weight).sum()
        ),
        "log_magnitude_error_noisy": float(log_noisy_error),
        "log_magnitude_error_enhanced": float(log_enhanced_error),
        "log_magnitude_error_reduction": float(
            log_noisy_error - log_enhanced_error
        ),
    }


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint["config"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("results/v13/error_analysis/per_file_analysis.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata/test.csv"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, action="append", default=None
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v13/listening_set"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    checkpoints = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    if len(checkpoints) != 3:
        parser.error("exactly three checkpoints are required")
    for path in [args.analysis, args.metadata, *checkpoints]:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(ROOT / path)

    frame = pd.read_csv(ROOT / args.analysis)
    selected = _select_cases(frame)
    metadata = pd.read_csv(ROOT / args.metadata)
    required_metadata = [
        "noisy_path",
        "clean_path",
        "sample_rate",
        "num_samples",
    ]
    missing_metadata = [
        column for column in required_metadata if column not in selected.columns
    ]
    if missing_metadata:
        selected = selected.merge(
            metadata[["file_id", *missing_metadata]],
            on="file_id",
            how="left",
        )
    if selected["noisy_path"].isna().any():
        raise ValueError("Selected files are missing from metadata")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selection_manifest.csv", index=False)
    dataset = PairedSpeechDataset(
        metadata_csv=ROOT / args.metadata,
        project_root=ROOT,
        sample_rate=16000,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    dataset_indices = {
        file_id: index for index, file_id in enumerate(metadata["file_id"])
    }
    device = torch.device(args.device)
    diagnostics: dict[str, dict[str, Any]] = {
        row["file_id"]: {
            "case_id": row["case_id"],
            "file_id": row["file_id"],
            "categories": row["categories"],
        }
        for _, row in selected.iterrows()
    }
    blind_rows: list[dict[str, str]] = []
    rng = np.random.default_rng(13_002)

    for seed_index, checkpoint_path in enumerate(checkpoints):
        model, config = _load_model(ROOT / checkpoint_path, device)
        seed = int(config["project"]["seed"])
        with torch.inference_mode():
            for _, row in selected.iterrows():
                file_id = row["file_id"]
                item = dataset[dataset_indices[file_id]]
                noisy = item["noisy"].unsqueeze(0).to(device)
                clean = item["clean"].unsqueeze(0).to(device)
                output = model(noisy)
                case_dir = output_dir / row["case_id"]
                if seed_index == 0:
                    _save_audio(case_dir / "noisy.wav", noisy, 16000)
                    _save_audio(case_dir / "clean.wav", clean, 16000)
                _save_audio(
                    case_dir / f"enhanced_seed{seed}.wav",
                    output.enhanced,
                    16000,
                )
                if seed_index == 0:
                    noisy_spectrum, _ = model._analysis(
                        noisy.squeeze(1), pad_end=True
                    )
                    clean_spectrum, _ = model._analysis(
                        clean.squeeze(1), pad_end=True
                    )
                    diagnostics[file_id].update(
                        _mask_diagnostics(
                            noisy_spectrum=noisy_spectrum,
                            clean_spectrum=clean_spectrum,
                            enhanced_spectrum=output.speech_spectrum,
                            mask=output.magnitude_mask,
                            sample_rate=16000,
                        )
                    )
                    _plot_diagnostic(
                        case_dir / "spectrogram_and_mask.png",
                        noisy_spectrum=noisy_spectrum,
                        clean_spectrum=clean_spectrum,
                        enhanced_spectrum=output.speech_spectrum,
                        mask=output.magnitude_mask,
                        sample_rate=16000,
                        hop_length=int(model.hop_length),
                        file_id=file_id,
                        categories=row["categories"],
                    )

                    blind_dir = output_dir / "blind" / row["case_id"]
                    _save_audio(blind_dir / "reference_clean.wav", clean, 16000)
                    enhanced_is_a = bool(rng.integers(0, 2))
                    a = output.enhanced if enhanced_is_a else noisy
                    b = noisy if enhanced_is_a else output.enhanced
                    _save_audio(blind_dir / "candidate_A.wav", a, 16000)
                    _save_audio(blind_dir / "candidate_B.wav", b, 16000)
                    blind_rows.append(
                        {
                            "case_id": row["case_id"],
                            "file_id": file_id,
                            "candidate_A": "enhanced" if enhanced_is_a else "noisy",
                            "candidate_B": "noisy" if enhanced_is_a else "enhanced",
                        }
                    )
        del model

    diagnostic_frame = pd.DataFrame(diagnostics.values())
    diagnostic_frame = selected.drop(
        columns=["noisy_path", "clean_path", "sample_rate", "num_samples"]
    ).merge(diagnostic_frame, on=["case_id", "file_id", "categories"])
    diagnostic_frame.to_csv(output_dir / "diagnostics.csv", index=False)
    pd.DataFrame(blind_rows).to_csv(output_dir / "blind_key.csv", index=False)
    record = {
        "purpose": "post-freeze diagnostic listening only",
        "standard_test_used_for_model_selection": False,
        "architecture_reselection_permitted": False,
        "selection_seed": 13002,
        "representative_audio_seed": 1200,
        "all_audio_seeds": [1200, 1201, 1202],
        "cases": len(selected),
        "categories": selected["categories"].str.split(";").explode().value_counts().to_dict(),
        "instructions": {
            "diagnostic": "Each CASE directory contains named noisy, clean, and three enhanced files plus a spectrogram/mask image.",
            "blind": "Listen to reference_clean.wav, then score candidate_A.wav and candidate_B.wav before opening blind_key.csv.",
        },
    }
    (output_dir / "README.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
