#!/usr/bin/env python3
"""Export CPU-only audio and mask diagnostics for V15 A/B collapse cases."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from cnvqg.data import PairedSpeechDataset
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    state = checkpoint.get(
        "ema_model_state_dict", checkpoint["model_state_dict"]
    )
    model.load_state_dict(state)
    return model.eval()


def _save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = waveform.squeeze().detach().cpu().numpy()
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        subtype="PCM_16",
    )


def _dataset(metadata: Path) -> tuple[PairedSpeechDataset, dict[str, int]]:
    frame = pd.read_csv(metadata)
    dataset = PairedSpeechDataset(
        metadata_csv=metadata,
        project_root=ROOT,
        sample_rate=16_000,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    return dataset, {
        str(file_id): index
        for index, file_id in enumerate(frame["file_id"])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path(
            "results/v15/preservation/quiet_level_identity_seed1200/"
            "error_analysis_vs_quiet_level/identity_collapse_cases.csv"
        ),
    )
    parser.add_argument(
        "--cross-metadata",
        type=Path,
        default=Path("data/processed/dns_cross_domain_dev/metadata.csv"),
    )
    parser.add_argument(
        "--voice-metadata",
        type=Path,
        default=Path(
            "data/processed/voicebank_demand/metadata/"
            "v12_architecture_selection_400.csv"
        ),
    )
    parser.add_argument(
        "--candidate-a-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v15/preservation/quiet_level_seed1200/"
            "epoch_003.pt"
        ),
    )
    parser.add_argument(
        "--candidate-b-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v15/preservation/quiet_level_identity_seed1200/"
            "epoch_003.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/v15/preservation/quiet_level_identity_seed1200/"
            "error_analysis_vs_quiet_level/listening_set"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    for path in (
        args.analysis,
        args.cross_metadata,
        args.voice_metadata,
        args.candidate_a_checkpoint,
        args.candidate_b_checkpoint,
    ):
        if not (ROOT / path).is_file():
            raise FileNotFoundError(ROOT / path)

    selection = pd.read_csv(ROOT / args.analysis)
    selection = selection[
        ["dataset", "file_id", "speaker_id", "duration_seconds"]
    ].copy()
    if selection.empty:
        raise ValueError("No identity-collapse cases were selected")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "selection_manifest.csv", index=False)

    datasets = {
        "cross_domain_dev": _dataset(ROOT / args.cross_metadata),
        "voicebank_dev400": _dataset(ROOT / args.voice_metadata),
    }
    items: dict[tuple[str, str], dict[str, torch.Tensor | str]] = {}
    for row in selection.itertuples(index=False):
        dataset, indices = datasets[row.dataset]
        item = dataset[indices[row.file_id]]
        items[(row.dataset, row.file_id)] = item
        case_dir = output_dir / row.dataset / row.file_id
        _save_audio(case_dir / "noisy.wav", item["noisy"], 16_000)
        _save_audio(case_dir / "clean.wav", item["clean"], 16_000)

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for label, checkpoint_path in (
        ("candidate_a", args.candidate_a_checkpoint),
        ("candidate_b", args.candidate_b_checkpoint),
    ):
        model = _load_model(ROOT / checkpoint_path, device)
        with torch.inference_mode():
            for row in selection.itertuples(index=False):
                item = items[(row.dataset, row.file_id)]
                noisy = item["noisy"].unsqueeze(0).to(device)
                output = model(noisy)
                enhanced = output.enhanced[..., : noisy.shape[-1]]
                mask = output.magnitude_mask.detach().float()
                residual = enhanced - noisy
                noisy_rms = noisy.square().mean().sqrt().clamp_min(1e-8)
                residual_rms = residual.square().mean().sqrt()
                case_dir = output_dir / row.dataset / row.file_id
                _save_audio(
                    case_dir / f"{label}.wav", enhanced, 16_000
                )
                rows.append(
                    {
                        "dataset": row.dataset,
                        "file_id": row.file_id,
                        "model": label,
                        "waveform_residual_rms": float(residual_rms),
                        "residual_to_noisy_rms": float(
                            residual_rms / noisy_rms
                        ),
                        "mask_mean": float(mask.mean()),
                        "mask_p10": float(torch.quantile(mask, 0.10)),
                        "mask_p50": float(torch.quantile(mask, 0.50)),
                        "mask_p90": float(torch.quantile(mask, 0.90)),
                        "mask_fraction_below_0_5": float(
                            (mask < 0.5).float().mean()
                        ),
                        "mask_fraction_below_0_8": float(
                            (mask < 0.8).float().mean()
                        ),
                        "mask_fraction_above_1_0": float(
                            (mask > 1.0).float().mean()
                        ),
                    }
                )
        del model
        gc.collect()

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output_dir / "model_diagnostics.csv", index=False)
    paired = diagnostics.pivot(
        index=["dataset", "file_id"],
        columns="model",
        values=[
            "waveform_residual_rms",
            "residual_to_noisy_rms",
            "mask_mean",
            "mask_p10",
            "mask_p50",
            "mask_p90",
            "mask_fraction_below_0_5",
            "mask_fraction_below_0_8",
            "mask_fraction_above_1_0",
        ],
    )
    paired.columns = [
        f"{metric}_{model}" for metric, model in paired.columns
    ]
    paired["b_to_a_waveform_residual_ratio"] = (
        paired["waveform_residual_rms_candidate_b"]
        / paired["waveform_residual_rms_candidate_a"].clip(lower=1e-8)
    )
    paired.reset_index().to_csv(
        output_dir / "paired_diagnostics.csv", index=False
    )
    print(
        paired[
            [
                "mask_mean_candidate_a",
                "mask_mean_candidate_b",
                "b_to_a_waveform_residual_ratio",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
