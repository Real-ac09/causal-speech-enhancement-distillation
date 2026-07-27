#!/usr/bin/env python3
"""CPU-only paired and gate analysis for V15 candidates A and D.

The saved development metrics are used for the primary paired comparison.
Candidate D is replayed only to inspect its causal gate.  On the small
cross-domain development set, the frozen candidate-A backbone is also
evaluated at predeclared constant residual strengths to estimate whether the
scalar blend has useful capacity that the learned gate failed to exploit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import compute_speech_metrics
from cnvqg.models.causal_confidence_gate_v14 import CausalConfidenceGateV14
from cnvqg.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("pesq", "si_sdr", "stoi", "estoi")
GATE_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_evaluation(directory: Path, prefix: str) -> pd.DataFrame:
    frame = (
        pd.read_csv(directory / "per_file_metrics.csv")
        .set_index("file_id")
        .sort_index()
    )
    return frame.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column != "speaker_id"
        }
    ).drop(columns=["speaker_id"])


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _trimmed_mean(values: np.ndarray, proportion: float = 0.05) -> float:
    ordered = np.sort(values)
    count = int(np.floor(len(ordered) * proportion))
    trimmed = ordered[count:-count] if count else ordered
    return float(trimmed.mean())


def _validate_and_join(
    *,
    metadata_path: Path,
    candidate_a: Path,
    candidate_d: Path,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path).set_index("file_id").sort_index()
    candidate_a_frame = _load_evaluation(candidate_a, "a")
    candidate_d_frame = _load_evaluation(candidate_d, "d")
    if not (
        metadata.index.equals(candidate_a_frame.index)
        and candidate_a_frame.index.equals(candidate_d_frame.index)
    ):
        raise ValueError(
            f"Metadata and evaluation IDs differ for {metadata_path}"
        )
    frame = metadata.join(candidate_a_frame).join(candidate_d_frame)
    for metric in METRICS:
        difference = (
            frame[f"a_noisy_{metric}"] - frame[f"d_noisy_{metric}"]
        ).abs()
        if float(difference.max()) > 1e-7:
            raise ValueError(
                f"Noisy {metric} values differ between candidates A and D"
            )
        noisy = frame[f"a_noisy_{metric}"]
        frame[f"a_{metric}_gain"] = frame[f"a_enhanced_{metric}"] - noisy
        frame[f"d_{metric}_gain"] = frame[f"d_enhanced_{metric}"] - noisy
        frame[f"d_minus_a_{metric}"] = (
            frame[f"d_enhanced_{metric}"]
            - frame[f"a_enhanced_{metric}"]
        )
        a_harm = frame[f"a_{metric}_gain"] < 0.0
        d_harm = frame[f"d_{metric}_gain"] < 0.0
        frame[f"{metric}_harm_transition"] = np.select(
            [
                a_harm & d_harm,
                a_harm & ~d_harm,
                ~a_harm & d_harm,
            ],
            ["persistent_harm", "repaired", "introduced"],
            default="persistent_non_harm",
        )
    frame["pesq_stoi_tradeoff"] = np.select(
        [
            (frame["d_minus_a_pesq"] > 0.0)
            & (frame["d_minus_a_stoi"] > 0.0),
            (frame["d_minus_a_pesq"] <= 0.0)
            & (frame["d_minus_a_stoi"] > 0.0),
            (frame["d_minus_a_pesq"] > 0.0)
            & (frame["d_minus_a_stoi"] <= 0.0),
        ],
        [
            "both_improved",
            "stoi_up_pesq_down",
            "pesq_up_stoi_down",
        ],
        default="both_regressed",
    )
    return frame


def _metric_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    delta = frame[f"d_minus_a_{metric}"].to_numpy(float)
    transition = frame[f"{metric}_harm_transition"]
    return {
        "candidate_a_enhanced_mean": float(
            frame[f"a_enhanced_{metric}"].mean()
        ),
        "candidate_d_enhanced_mean": float(
            frame[f"d_enhanced_{metric}"].mean()
        ),
        "d_minus_a_mean": float(delta.mean()),
        "d_minus_a_ci95": _bootstrap_ci(
            delta, samples=samples, rng=rng
        ),
        "d_minus_a_median": float(np.median(delta)),
        "d_minus_a_5pct_trimmed_mean": _trimmed_mean(delta),
        "d_win_rate": float((delta > 0.0).mean()),
        "candidate_a_harm_rate": float(
            (frame[f"a_{metric}_gain"] < 0.0).mean()
        ),
        "candidate_d_harm_rate": float(
            (frame[f"d_{metric}_gain"] < 0.0).mean()
        ),
        "repaired_items": int((transition == "repaired").sum()),
        "introduced_harm_items": int((transition == "introduced").sum()),
    }


def _dataset_summary(
    frame: pd.DataFrame,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "items": int(len(frame)),
        "metrics": {
            metric: _metric_summary(
                frame,
                metric=metric,
                samples=samples,
                rng=rng,
            )
            for metric in METRICS
        },
        "pesq_stoi_tradeoff_counts": {
            str(key): int(value)
            for key, value in frame["pesq_stoi_tradeoff"]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def _group_summary(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_column in group_columns:
        for value, group in frame.groupby(group_column, sort=True):
            row: dict[str, Any] = {
                "condition": group_column,
                "value": value,
                "items": int(len(group)),
            }
            for metric in METRICS:
                delta = group[f"d_minus_a_{metric}"]
                row[f"d_minus_a_{metric}_mean"] = float(delta.mean())
                row[f"d_{metric}_win_rate"] = float((delta > 0.0).mean())
                row[f"a_{metric}_harm_rate"] = float(
                    (group[f"a_{metric}_gain"] < 0.0).mean()
                )
                row[f"d_{metric}_harm_rate"] = float(
                    (group[f"d_{metric}_gain"] < 0.0).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _ranked_cases(
    frames: dict[str, pd.DataFrame],
    *,
    largest: bool,
    count: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in frames.items():
        for metric in METRICS:
            ranked = frame[f"d_minus_a_{metric}"].sort_values(
                ascending=not largest
            )
            for rank, (file_id, value) in enumerate(
                ranked.iloc[:count].items(), start=1
            ):
                source = frame.loc[file_id]
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "rank": rank,
                        "file_id": file_id,
                        "speaker_id": source["speaker_id"],
                        "duration_seconds": source["duration_seconds"],
                        "d_minus_a": float(value),
                        "noisy": float(source[f"a_noisy_{metric}"]),
                        "candidate_a": float(
                            source[f"a_enhanced_{metric}"]
                        ),
                        "candidate_d": float(
                            source[f"d_enhanced_{metric}"]
                        ),
                        "candidate_a_gain": float(
                            source[f"a_{metric}_gain"]
                        ),
                        "candidate_d_gain": float(
                            source[f"d_{metric}_gain"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _load_model(checkpoint_path: Path) -> CausalConfidenceGateV14:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not isinstance(model, CausalConfidenceGateV14):
        raise TypeError(
            "Candidate D checkpoint is not CausalConfidenceGateV14"
        )
    return model


def _tensor_to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return waveform.squeeze().detach().cpu().numpy().astype(np.float32)


def _gate_statistics(
    strength: torch.Tensor,
    summaries: torch.Tensor,
) -> dict[str, float]:
    strength = strength.detach().float().flatten()
    summaries = summaries.detach().float().reshape(-1, 4)
    result = {
        "gate_strength_mean": float(strength.mean()),
        "gate_strength_std": float(strength.std(unbiased=False)),
        "gate_strength_min": float(strength.min()),
        "gate_strength_max": float(strength.max()),
        "gate_temporal_abs_delta_mean": float(
            (strength[1:] - strength[:-1]).abs().mean()
        )
        if strength.numel() > 1
        else 0.0,
    }
    for quantile in GATE_QUANTILES:
        label = f"{int(round(100 * quantile)):02d}"
        result[f"gate_strength_p{label}"] = float(
            torch.quantile(strength, quantile)
        )
    for threshold, label in (
        (0.50, "050"),
        (0.75, "075"),
        (0.90, "090"),
        (0.95, "095"),
        (0.99, "099"),
    ):
        result[f"gate_fraction_below_{label}"] = float(
            (strength < threshold).float().mean()
        )
    for index, name in enumerate(
        ("mixture_log_rms", "mixture_flatness", "mixture_high_ratio", "mixture_flux")
    ):
        result[f"{name}_mean"] = float(summaries[:, index].mean())
        result[f"{name}_std"] = float(
            summaries[:, index].std(unbiased=False)
        )
    return result


def _strength_output(
    model: CausalConfidenceGateV14,
    noisy: torch.Tensor,
    mixture_spectrum: torch.Tensor,
    base_mask: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    magnitude_mask = 1.0 + float(strength) * (base_mask - 1.0)
    speech_spectrum = magnitude_mask * mixture_spectrum
    return model.backbone._synthesis(
        speech_spectrum,
        noisy.shape[-1],
    ).unsqueeze(1)


def _infer_gate_dataset(
    *,
    model: CausalConfidenceGateV14,
    metadata_path: Path,
    dataset_name: str,
    output_path: Path,
    max_items: int | None,
    oracle_strengths: tuple[float, ...] | None,
    oracle_output_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    dataset = PairedSpeechDataset(
        metadata_csv=metadata_path,
        project_root=ROOT,
        sample_rate=16_000,
        chunk_seconds=None,
        random_crop=False,
        peak_normalize=False,
    )
    item_count = len(dataset)
    if max_items is not None:
        item_count = min(item_count, max_items)
    gate_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in tqdm(
            range(item_count),
            desc=f"Gate diagnostics: {dataset_name}",
        ):
            item = dataset[index]
            noisy = item["noisy"].unsqueeze(0)
            clean = item["clean"]
            base = model.backbone(noisy)
            mixture_spectrum = base.speech_spectrum + base.noise_spectrum
            strength = model.confidence_gate(
                base.continuous_noise_state,
                mixture_spectrum,
            )
            summaries = model.confidence_gate.mixture_summaries(
                mixture_spectrum
            )
            metadata_row = metadata.iloc[index]
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "file_id": item["file_id"],
                "speaker_id": item["speaker_id"],
                "duration_seconds": float(
                    metadata_row["duration_seconds"]
                ),
                **_gate_statistics(strength, summaries),
            }
            for column in (
                "target_snr_db",
                "actual_snr_db",
                "target_clean_rms_dbfs",
                "actual_clean_rms_dbfs",
            ):
                if column in metadata.columns:
                    row[column] = float(metadata_row[column])
            gate_rows.append(row)

            if oracle_strengths is not None:
                noisy_np = _tensor_to_numpy(noisy)
                clean_np = _tensor_to_numpy(clean)
                for fixed_strength in oracle_strengths:
                    enhanced = _strength_output(
                        model,
                        noisy,
                        mixture_spectrum,
                        base.magnitude_mask,
                        fixed_strength,
                    )
                    metrics = compute_speech_metrics(
                        noisy=noisy_np,
                        enhanced=_tensor_to_numpy(enhanced),
                        clean=clean_np,
                        sample_rate=16_000,
                    )
                    oracle_rows.append(
                        {
                            "dataset": dataset_name,
                            "file_id": item["file_id"],
                            "strength": float(fixed_strength),
                            **metrics,
                        }
                    )
            pd.DataFrame(gate_rows).to_csv(output_path, index=False)
            if oracle_output_path is not None and oracle_rows:
                pd.DataFrame(oracle_rows).to_csv(
                    oracle_output_path,
                    index=False,
                )
    return pd.DataFrame(gate_rows), pd.DataFrame(oracle_rows)


def _gate_summary(
    paired: pd.DataFrame,
    gate: pd.DataFrame,
) -> dict[str, Any]:
    numeric_gate_columns = [
        column
        for column in gate.columns
        if column.startswith("gate_") or column.startswith("mixture_")
    ]
    joined = paired.join(
        gate.set_index("file_id")[numeric_gate_columns],
        how="inner",
    )
    correlations: dict[str, dict[str, float | None]] = {}
    targets = [
        *(f"d_minus_a_{metric}" for metric in METRICS),
        *(f"a_{metric}_gain" for metric in METRICS),
    ]
    for gate_column in numeric_gate_columns:
        correlations[gate_column] = {}
        for target in targets:
            value = joined[[gate_column, target]].corr(
                method="spearman"
            ).iloc[0, 1]
            correlations[gate_column][target] = (
                None if pd.isna(value) else float(value)
            )
    aggregate = {
        column: {
            "mean": float(gate[column].mean()),
            "std_across_files": float(
                gate[column].std(ddof=0)
            ),
            "minimum": float(gate[column].min()),
            "maximum": float(gate[column].max()),
        }
        for column in numeric_gate_columns
    }
    return {
        "items": int(len(gate)),
        "aggregate": aggregate,
        "spearman_correlations": correlations,
    }


def _oracle_summary(
    oracle: pd.DataFrame,
    candidate_a: pd.DataFrame,
) -> dict[str, Any]:
    if oracle.empty:
        return {"items": 0, "strengths": {}, "per_file_oracle": {}}
    strength_rows: dict[str, Any] = {}
    for strength, group in oracle.groupby("strength", sort=True):
        strength_rows[f"{float(strength):.6g}"] = {
            "items": int(len(group)),
            "pesq_mean": float(group["enhanced_pesq"].mean()),
            "si_sdr_mean": float(group["enhanced_si_sdr"].mean()),
            "stoi_mean": float(group["enhanced_stoi"].mean()),
            "estoi_mean": float(group["enhanced_estoi"].mean()),
            "pesq_gain_mean": float(
                (group["enhanced_pesq"] - group["noisy_pesq"]).mean()
            ),
            "si_sdr_gain_mean": float(
                (
                    group["enhanced_si_sdr"]
                    - group["noisy_si_sdr"]
                ).mean()
            ),
            "stoi_gain_mean": float(
                (group["enhanced_stoi"] - group["noisy_stoi"]).mean()
            ),
            "estoi_gain_mean": float(
                (group["enhanced_estoi"] - group["noisy_estoi"]).mean()
            ),
            "stoi_harm_rate": float(
                (
                    group["enhanced_stoi"]
                    < group["noisy_stoi"] - 1e-6
                ).mean()
            ),
        }
    oracle_indices = oracle.groupby("file_id")["enhanced_stoi"].idxmax()
    best = oracle.loc[oracle_indices].copy()
    result = {
        "items": int(oracle["file_id"].nunique()),
        "strengths": strength_rows,
        "per_file_oracle": {
            "selection_rule": (
                "highest STOI among the fixed-strength grid; descriptive "
                "upper bound using clean references, not a deployable result"
            ),
            "selected_strength_counts": {
                f"{float(key):.6g}": int(value)
                for key, value in best["strength"]
                .value_counts()
                .sort_index()
                .items()
            },
            "pesq_mean": float(best["enhanced_pesq"].mean()),
            "si_sdr_mean": float(best["enhanced_si_sdr"].mean()),
            "stoi_mean": float(best["enhanced_stoi"].mean()),
            "estoi_mean": float(best["enhanced_estoi"].mean()),
            "pesq_gain_mean": float(
                (best["enhanced_pesq"] - best["noisy_pesq"]).mean()
            ),
            "si_sdr_gain_mean": float(
                (
                    best["enhanced_si_sdr"]
                    - best["noisy_si_sdr"]
                ).mean()
            ),
            "stoi_gain_mean": float(
                (best["enhanced_stoi"] - best["noisy_stoi"]).mean()
            ),
            "estoi_gain_mean": float(
                (best["enhanced_estoi"] - best["noisy_estoi"]).mean()
            ),
            "stoi_harm_rate": float(
                (
                    best["enhanced_stoi"]
                    < best["noisy_stoi"] - 1e-6
                ).mean()
            ),
        },
    }
    result["stoi_harm_tolerance"] = 1e-6
    if 1.0 in set(oracle["strength"]):
        strength_one = oracle[oracle["strength"] == 1.0].set_index(
            "file_id"
        )
        aligned = candidate_a.join(
            strength_one[
                [
                    "enhanced_pesq",
                    "enhanced_si_sdr",
                    "enhanced_stoi",
                    "enhanced_estoi",
                ]
            ],
            how="inner",
        )
        result["strength_one_reproduction_audit"] = {
            metric: {
                "mean_recomputed_minus_saved_a": float(
                    (
                        aligned[f"enhanced_{metric}"]
                        - aligned[f"a_enhanced_{metric}"]
                    ).mean()
                ),
                "maximum_absolute_difference": float(
                    (
                        aligned[f"enhanced_{metric}"]
                        - aligned[f"a_enhanced_{metric}"]
                    ).abs().max()
                ),
            }
            for metric in METRICS
        }
    return result


def _parse_strengths(raw: str) -> tuple[float, ...]:
    strengths = tuple(float(value.strip()) for value in raw.split(","))
    if not strengths:
        raise ValueError("At least one oracle strength is required")
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("Oracle strengths must be within [0, 1]")
    if len(set(strengths)) != len(strengths):
        raise ValueError("Oracle strengths must be unique")
    return tuple(sorted(strengths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--candidate-a-root",
        type=Path,
        default=Path("results/v15/preservation/quiet_level_seed1200"),
    )
    parser.add_argument(
        "--candidate-d-root",
        type=Path,
        default=Path(
            "results/v15/preservation/"
            "causal_preservation_gate_seed1200"
        ),
    )
    parser.add_argument(
        "--candidate-d-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v15/preservation/"
            "causal_preservation_gate_seed1200/epoch_003.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/v15/preservation/"
            "causal_preservation_gate_seed1200/"
            "error_analysis_vs_quiet_level"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=15_041)
    parser.add_argument(
        "--oracle-strengths",
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated constant residual-strength grid.",
    )
    parser.add_argument("--max-cross-items", type=int)
    parser.add_argument("--max-voice-items", type=int)
    parser.add_argument("--skip-gate-inference", action="store_true")
    parser.add_argument("--skip-oracle-sweep", action="store_true")
    parser.add_argument(
        "--voice-strength-sweep",
        action="store_true",
        help=(
            "Also run the fixed-strength metric sweep on VoiceBank. This is "
            "slower and is intended for the final post-mortem only."
        ),
    )
    parser.add_argument(
        "--reuse-cross-inference",
        action="store_true",
        help=(
            "Reuse complete cross gate/sweep CSVs already in output-dir "
            "while running additional VoiceBank diagnostics."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    args = parser.parse_args()

    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    strengths = _parse_strengths(args.oracle_strengths)

    cross_metadata = _resolve(args.cross_metadata)
    voice_metadata = _resolve(args.voice_metadata)
    candidate_a_root = _resolve(args.candidate_a_root)
    candidate_d_root = _resolve(args.candidate_d_root)
    checkpoint_path = _resolve(args.candidate_d_checkpoint)
    output_dir = _resolve(args.output_dir)
    required = (
        cross_metadata,
        voice_metadata,
        candidate_a_root / "cross_domain_dev/per_file_metrics.csv",
        candidate_a_root / "voicebank_dev400/per_file_metrics.csv",
        candidate_d_root / "cross_domain_dev/per_file_metrics.csv",
        candidate_d_root / "voicebank_dev400/per_file_metrics.csv",
    )
    if not args.skip_gate_inference:
        required = (*required, checkpoint_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    cross = _validate_and_join(
        metadata_path=cross_metadata,
        candidate_a=candidate_a_root / "cross_domain_dev",
        candidate_d=candidate_d_root / "cross_domain_dev",
    )
    voice = _validate_and_join(
        metadata_path=voice_metadata,
        candidate_a=candidate_a_root / "voicebank_dev400",
        candidate_d=candidate_d_root / "voicebank_dev400",
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    report: dict[str, Any] = {
        "status": "complete",
        "analysis_role": (
            "paired development post-mortem and CPU-only gate diagnostic"
        ),
        "external_test_used": False,
        "candidate_a": "v15_quiet_level_seed1200_epoch3",
        "candidate_d": "v15_causal_preservation_gate_seed1200_epoch3",
        "bootstrap": {
            "method": "paired_file",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "datasets": {
            "cross_domain_dev": _dataset_summary(
                cross,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
            "voicebank_dev400": _dataset_summary(
                voice,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
        },
        "gate_diagnostics": {},
        "constant_strength_cross_domain_sweep": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cross.reset_index().to_csv(
        output_dir / "cross_per_file.csv", index=False
    )
    voice.reset_index().to_csv(
        output_dir / "voice_per_file.csv", index=False
    )
    _group_summary(
        cross,
        group_columns=("target_clean_rms_dbfs", "target_snr_db"),
    ).to_csv(output_dir / "cross_condition_summary.csv", index=False)
    _group_summary(
        voice,
        group_columns=("speaker_id",),
    ).to_csv(output_dir / "voice_speaker_summary.csv", index=False)
    frames = {"cross_domain_dev": cross, "voicebank_dev400": voice}
    _ranked_cases(frames, largest=False).to_csv(
        output_dir / "top_regressions.csv", index=False
    )
    _ranked_cases(frames, largest=True).to_csv(
        output_dir / "top_improvements.csv", index=False
    )

    if not args.skip_gate_inference:
        model = _load_model(checkpoint_path)
        cross_gate_path = output_dir / "cross_gate_diagnostics.csv"
        cross_sweep_path = (
            output_dir / "cross_constant_strength_sweep.csv"
        )
        if args.reuse_cross_inference:
            if not cross_gate_path.is_file():
                raise FileNotFoundError(cross_gate_path)
            cross_gate = pd.read_csv(cross_gate_path)
            if args.skip_oracle_sweep:
                oracle = pd.DataFrame()
            else:
                if not cross_sweep_path.is_file():
                    raise FileNotFoundError(cross_sweep_path)
                oracle = pd.read_csv(cross_sweep_path)
        else:
            cross_gate, oracle = _infer_gate_dataset(
                model=model,
                metadata_path=cross_metadata,
                dataset_name="cross_domain_dev",
                output_path=cross_gate_path,
                max_items=args.max_cross_items,
                oracle_strengths=(
                    None if args.skip_oracle_sweep else strengths
                ),
                oracle_output_path=(
                    None if args.skip_oracle_sweep else cross_sweep_path
                ),
            )
        voice_sweep_path = (
            output_dir / "voice_constant_strength_sweep.csv"
        )
        voice_gate, voice_oracle = _infer_gate_dataset(
            model=model,
            metadata_path=voice_metadata,
            dataset_name="voicebank_dev400",
            output_path=output_dir / "voice_gate_diagnostics.csv",
            max_items=args.max_voice_items,
            oracle_strengths=(
                strengths if args.voice_strength_sweep else None
            ),
            oracle_output_path=(
                voice_sweep_path if args.voice_strength_sweep else None
            ),
        )
        report["gate_diagnostics"] = {
            "checkpoint": str(args.candidate_d_checkpoint),
            "device": "cpu",
            "threads": args.threads,
            "cross_domain_dev": _gate_summary(cross, cross_gate),
            "voicebank_dev400": _gate_summary(voice, voice_gate),
        }
        report["constant_strength_cross_domain_sweep"] = _oracle_summary(
            oracle,
            cross,
        )
        if args.voice_strength_sweep:
            report["constant_strength_voicebank_sweep"] = (
                _oracle_summary(voice_oracle, voice)
            )

    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output_dir / "summary.json"),
                "external_test_used": report["external_test_used"],
                "cross_items": report["datasets"][
                    "cross_domain_dev"
                ]["items"],
                "voice_items": report["datasets"][
                    "voicebank_dev400"
                ]["items"],
                "gate_inference": not args.skip_gate_inference,
                "oracle_sweep": (
                    not args.skip_gate_inference
                    and not args.skip_oracle_sweep
                ),
                "voice_strength_sweep": args.voice_strength_sweep,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
