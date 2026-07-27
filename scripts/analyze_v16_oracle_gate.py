#!/usr/bin/env python3
"""Run CPU-only postmortem diagnostics for the frozen V16 oracle gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
GRID = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
METRICS = ("pesq", "si_sdr", "stoi", "estoi")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _finite_correlation(first: pd.Series, second: pd.Series) -> float | None:
    values = pd.concat([first, second], axis=1).dropna().to_numpy(float)
    if len(values) < 2:
        return None
    if np.std(values[:, 0]) == 0.0 or np.std(values[:, 1]) == 0.0:
        return None
    return float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])


def _gate_diagnostics(
    *,
    model: torch.nn.Module,
    metadata_path: Path,
    description: str,
) -> pd.DataFrame:
    dataset = PairedSpeechDataset(
        metadata_csv=metadata_path,
        project_root=ROOT,
        sample_rate=16_000,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    rows = []
    with torch.inference_mode():
        for index in tqdm(
            range(len(dataset)),
            desc=f"V16 gate {description}",
        ):
            item = dataset[index]
            noisy = item["noisy"].unsqueeze(0)
            output = model(noisy)
            strengths = (
                output.gate_strength.detach().cpu().numpy().reshape(-1)
            )
            waveform = noisy.numpy().reshape(-1).astype(np.float64)
            rms = float(np.sqrt(np.mean(np.square(waveform)) + 1e-12))
            rows.append(
                {
                    "file_id": item["file_id"],
                    "speaker_id": item["speaker_id"],
                    "frames": int(len(strengths)),
                    "gate_mean": float(np.mean(strengths)),
                    "gate_std": float(np.std(strengths)),
                    "gate_min": float(np.min(strengths)),
                    "gate_max": float(np.max(strengths)),
                    "gate_fraction_below_0_99": float(
                        np.mean(strengths < 0.99)
                    ),
                    "gate_fraction_below_0_90": float(
                        np.mean(strengths < 0.90)
                    ),
                    "gate_fraction_below_0_75": float(
                        np.mean(strengths < 0.75)
                    ),
                    "noisy_rms_dbfs": float(20.0 * np.log10(rms)),
                }
            )
    return pd.DataFrame(rows)


def _gate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    weights = frame["frames"].to_numpy(float)
    return {
        "items": int(len(frame)),
        "mean_file_strength": float(frame["gate_mean"].mean()),
        "sd_file_strength": float(frame["gate_mean"].std(ddof=0)),
        "mean_within_file_strength_sd": float(frame["gate_std"].mean()),
        "minimum_frame_strength": float(frame["gate_min"].min()),
        "maximum_frame_strength": float(frame["gate_max"].max()),
        "fraction_frames_below_0_99": float(
            np.average(frame["gate_fraction_below_0_99"], weights=weights)
        ),
        "fraction_frames_below_0_90": float(
            np.average(frame["gate_fraction_below_0_90"], weights=weights)
        ),
        "fraction_frames_below_0_75": float(
            np.average(frame["gate_fraction_below_0_75"], weights=weights)
        ),
    }


def _strength_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        f"{float(key):.2f}": int(value)
        for key, value in frame["oracle_strength"]
        .value_counts()
        .sort_index()
        .items()
    }


def _label_summary(metadata: pd.DataFrame) -> dict[str, Any]:
    domains = {}
    for domain, group in metadata.groupby("domain"):
        domain_summary: dict[str, Any] = {
            "items": int(len(group)),
            "mean_strength": float(group["oracle_strength"].mean()),
            "full_strength_fraction": float(
                (group["oracle_strength"] == 1.0).mean()
            ),
            "strength_counts": _strength_counts(group),
            "fallback_fraction": float(
                (~group["oracle_feasible"].astype(bool)).mean()
            ),
        }
        domain_summary["conditions"] = {}
        for column in ("target_snr_db", "target_clean_rms_dbfs"):
            if column not in group.columns:
                continue
            conditions = group.dropna(subset=[column])
            if conditions.empty:
                continue
            domain_summary["conditions"][column] = {
                f"{float(value):.1f}": {
                    "items": int(len(condition)),
                    "target_mean": float(
                        condition["oracle_strength"].mean()
                    ),
                    "full_strength_fraction": float(
                        (condition["oracle_strength"] == 1.0).mean()
                    ),
                    "strength_at_most_0_5_fraction": float(
                        (condition["oracle_strength"] <= 0.5).mean()
                    ),
                }
                for value, condition in conditions.groupby(column)
            }
        domains[str(domain)] = domain_summary
    return {
        "items": int(len(metadata)),
        "mean_strength": float(metadata["oracle_strength"].mean()),
        "full_strength_fraction": float(
            (metadata["oracle_strength"] == 1.0).mean()
        ),
        "strength_counts": _strength_counts(metadata),
        "domains": domains,
    }


def _calibration_analysis(
    labelled: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = labelled.merge(
        diagnostics,
        on=["file_id", "speaker_id"],
        validate="one_to_one",
    )
    merged["gate_error"] = merged["gate_mean"] - merged["oracle_strength"]
    merged["gate_absolute_error"] = merged["gate_error"].abs()
    predicted = merged["gate_mean"].to_numpy()[:, None]
    merged["nearest_grid_strength"] = GRID[
        np.abs(predicted - GRID[None, :]).argmin(axis=1)
    ]
    by_target = {}
    for target, group in merged.groupby("oracle_strength"):
        by_target[f"{float(target):.2f}"] = {
            "items": int(len(group)),
            "predicted_mean": float(group["gate_mean"].mean()),
            "predicted_sd": float(group["gate_mean"].std(ddof=0)),
            "mae": float(group["gate_absolute_error"].mean()),
            "nearest_grid_accuracy": float(
                (group["nearest_grid_strength"] == target).mean()
            ),
        }
    by_domain = {}
    for domain, group in merged.groupby("domain"):
        by_domain[str(domain)] = {
            "items": int(len(group)),
            "target_mean": float(group["oracle_strength"].mean()),
            "predicted_mean": float(group["gate_mean"].mean()),
            "mae": float(group["gate_absolute_error"].mean()),
            "target_prediction_correlation": _finite_correlation(
                group["oracle_strength"],
                group["gate_mean"],
            ),
        }
    summary = {
        "mae": float(merged["gate_absolute_error"].mean()),
        "rmse": float(np.sqrt(np.mean(np.square(merged["gate_error"])))),
        "target_prediction_correlation": _finite_correlation(
            merged["oracle_strength"],
            merged["gate_mean"],
        ),
        "nearest_grid_accuracy": float(
            (
                merged["nearest_grid_strength"]
                == merged["oracle_strength"]
            ).mean()
        ),
        "predicted_strength_range": [
            float(merged["gate_mean"].min()),
            float(merged["gate_mean"].max()),
        ],
        "by_target": by_target,
        "by_domain": by_domain,
    }
    summary["macro_target_mae"] = float(
        np.mean([values["mae"] for values in by_target.values()])
    )
    summary["macro_nearest_grid_accuracy"] = float(
        np.mean(
            [
                values["nearest_grid_accuracy"]
                for values in by_target.values()
            ]
        )
    )
    return merged, summary


def _development_analysis(
    *,
    diagnostics: pd.DataFrame,
    metrics_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = pd.read_csv(metrics_path)
    metadata = pd.read_csv(metadata_path)
    keep = [
        column
        for column in (
            "file_id",
            "target_snr_db",
            "actual_snr_db",
            "target_clean_rms_dbfs",
            "actual_clean_rms_dbfs",
        )
        if column in metadata.columns
    ]
    merged = metrics.merge(
        diagnostics,
        on=["file_id", "speaker_id"],
        validate="one_to_one",
    )
    if len(keep) > 1:
        merged = merged.merge(
            metadata[keep],
            on="file_id",
            validate="one_to_one",
        )
    for metric in METRICS:
        merged[f"{metric}_gain"] = (
            merged[f"enhanced_{metric}"] - merged[f"noisy_{metric}"]
        )
    summary: dict[str, Any] = {
        "gate": _gate_summary(diagnostics),
        "correlations": {},
        "stoi_harmed_items": int((merged["stoi_gain"] < 0.0).sum()),
        "stoi_harm_rate": float((merged["stoi_gain"] < 0.0).mean()),
        "mean_gate_stoi_harmed": float(
            merged.loc[merged["stoi_gain"] < 0.0, "gate_mean"].mean()
        ),
        "mean_gate_stoi_not_harmed": float(
            merged.loc[merged["stoi_gain"] >= 0.0, "gate_mean"].mean()
        ),
    }
    for name in (
        "noisy_si_sdr",
        "noisy_pesq",
        "noisy_stoi",
        "noisy_estoi",
        "noisy_rms_dbfs",
        "pesq_gain",
        "si_sdr_gain",
        "stoi_gain",
        "estoi_gain",
    ):
        summary["correlations"][f"gate_mean_vs_{name}"] = (
            _finite_correlation(merged["gate_mean"], merged[name])
        )
    summary["conditions"] = {}
    for column in ("target_snr_db", "target_clean_rms_dbfs"):
        if column not in merged.columns:
            continue
        summary["conditions"][column] = {
            f"{float(value):.1f}": {
                "items": int(len(group)),
                "gate_mean": float(group["gate_mean"].mean()),
                "stoi_gain": float(group["stoi_gain"].mean()),
                "stoi_harm_rate": float((group["stoi_gain"] < 0.0).mean()),
            }
            for value, group in merged.groupby(column)
        }
    return merged, summary


def _voice_padding_analysis(
    selection_path: Path,
    labelled_train: pd.DataFrame,
    labelled_calibration: pd.DataFrame,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    sources = pd.DataFrame(selection["voice_items"])[
        ["file_id", "duration_seconds", "oracle_role"]
    ].rename(
        columns={
            "file_id": "source_file_id",
            "duration_seconds": "source_duration_seconds",
        }
    )
    labelled = pd.concat(
        [labelled_train, labelled_calibration],
        ignore_index=True,
    )
    voice = labelled[labelled["domain"] == "voicebank_demand"].merge(
        sources,
        on=["source_file_id", "oracle_role"],
        validate="one_to_one",
    )
    voice["padded_fraction"] = (
        (4.0 - voice["source_duration_seconds"]).clip(lower=0.0) / 4.0
    )
    padded_frame_fraction = float(voice["padded_fraction"].mean())
    full = voice["oracle_strength"] == 1.0
    return {
        "items": int(len(voice)),
        "mean_padded_fraction": padded_frame_fraction,
        "estimated_padded_frames_with_full_target_fraction_all_frames": float(
            (
                voice.loc[full, "padded_fraction"].sum()
                / len(voice)
            )
        ),
        "padded_fraction_by_oracle_strength": {
            f"{float(target):.2f}": float(group["padded_fraction"].mean())
            for target, group in voice.groupby("oracle_strength")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/v16/oracle_gate_seed16040/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/v16/oracle_gate_seed16040/postmortem"
        ),
    )
    args = parser.parse_args()
    checkpoint_path = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = _load_model(checkpoint_path)

    paths = {
        "train": Path("results/v16/oracle_labels/train/metadata.csv"),
        "calibration": Path(
            "results/v16/oracle_labels/calibration/metadata.csv"
        ),
        "cross": Path("data/processed/dns_cross_domain_dev/metadata.csv"),
        "voice": Path(
            "data/processed/voicebank_demand/metadata/"
            "v12_architecture_selection_400.csv"
        ),
    }
    labelled_train = pd.read_csv(_resolve(paths["train"]))
    labelled_calibration = pd.read_csv(_resolve(paths["calibration"]))

    diagnostic_frames = {}
    for name in ("calibration", "cross", "voice"):
        diagnostic_path = output_dir / f"{name}_gate_diagnostics.csv"
        if diagnostic_path.is_file():
            frame = pd.read_csv(diagnostic_path)
        else:
            frame = _gate_diagnostics(
                model=model,
                metadata_path=_resolve(paths[name]),
                description=name,
            )
            frame.to_csv(diagnostic_path, index=False)
        diagnostic_frames[name] = frame

    calibration, calibration_summary = _calibration_analysis(
        labelled_calibration,
        diagnostic_frames["calibration"],
    )
    calibration.to_csv(
        output_dir / "calibration_predictions.csv",
        index=False,
    )
    cross, cross_summary = _development_analysis(
        diagnostics=diagnostic_frames["cross"],
        metrics_path=ROOT
        / "results/v16/oracle_gate_seed16040/cross_domain_dev/"
        "per_file_metrics.csv",
        metadata_path=_resolve(paths["cross"]),
    )
    voice, voice_summary = _development_analysis(
        diagnostics=diagnostic_frames["voice"],
        metrics_path=ROOT
        / "results/v16/oracle_gate_seed16040/voicebank_dev400/"
        "per_file_metrics.csv",
        metadata_path=_resolve(paths["voice"]),
    )
    cross.to_csv(output_dir / "cross_per_file.csv", index=False)
    voice.to_csv(output_dir / "voice_per_file.csv", index=False)
    cross.sort_values("stoi_gain").head(20).to_csv(
        output_dir / "cross_worst_stoi.csv",
        index=False,
    )

    train_summary = _label_summary(labelled_train)
    calibration_label_summary = _label_summary(labelled_calibration)
    domain_means = [
        values["mean_strength"]
        for values in train_summary["domains"].values()
    ]
    domain_full_fractions = [
        values["full_strength_fraction"]
        for values in train_summary["domains"].values()
    ]
    summary = {
        "status": "complete",
        "role": "post_hoc_development_diagnosis",
        "external_test_used": False,
        "checkpoint": str(args.checkpoint),
        "training_labels": train_summary,
        "calibration_labels": calibration_label_summary,
        "equal_domain_sampling_expected_target_mean": float(
            np.mean(domain_means)
        ),
        "equal_domain_sampling_expected_full_strength_fraction": float(
            np.mean(domain_full_fractions)
        ),
        "calibration_prediction": calibration_summary,
        "voice_padding": _voice_padding_analysis(
            ROOT / "configs/v16/oracle_corpus_selection.json",
            labelled_train,
            labelled_calibration,
        ),
        "cross_domain": cross_summary,
        "voicebank": voice_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
