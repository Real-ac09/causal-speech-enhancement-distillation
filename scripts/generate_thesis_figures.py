#!/usr/bin/env python3
"""Generate the frozen dissertation performance and training figures.

The script deliberately separates final test evidence from development-only
controller studies and historical exploratory evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cnvqg-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cnvqg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "thesis"

BLUE = "#3066BE"
ORANGE = "#E07A2D"
GREEN = "#2A9D6F"
RED = "#C44536"
PURPLE = "#7251B5"
GREY = "#777777"
LIGHT_GREY = "#D7DCE2"

METRICS = {
    "enhanced_pesq": ("PESQ", "PESQ"),
    "enhanced_si_sdr": ("SI-SDR", "dB"),
    "enhanced_stoi": ("STOI", "Score"),
    "enhanced_estoi": ("ESTOI", "Score"),
}
NOISY_KEYS = {
    "enhanced_pesq": "noisy_pesq",
    "enhanced_si_sdr": "noisy_si_sdr",
    "enhanced_stoi": "noisy_stoi",
    "enhanced_estoi": "noisy_estoi",
}

PROVENANCE: list[dict[str, Any]] = []


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def record(
    figure: str,
    evidence_tier: str,
    dataset: str,
    model: str,
    metric: str,
    value: float,
    source: str,
    note: str = "",
) -> None:
    PROVENANCE.append(
        {
            "figure": figure,
            "evidence_tier": evidence_tier,
            "dataset": dataset,
            "model": model,
            "metric": metric,
            "value": value,
            "source": source,
            "note": note,
        }
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.alpha": 0.24,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def final_paths(dataset: str) -> tuple[str, str, str]:
    base = f"results/v14/replication_evaluation/{dataset}"
    return (
        f"{base}/v13_aggregate.json",
        f"{base}/v14_2_aggregate.json",
        f"{base}/v14_2_vs_v13_paired_hierarchical.json",
    )


def plot_final_absolute(output: Path, dataset: str) -> None:
    v13_path, v14_path, _ = final_paths(dataset)
    v13 = read_json(v13_path)
    v14 = read_json(v14_path)
    dataset_label = (
        "VoiceBank-DEMAND standard test"
        if dataset == "standard"
        else "DNS1 independent external set"
    )
    figure_name = f"01_final_{dataset}_absolute"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.ravel()
    models = ["Noisy", "V13", "V14.2"]
    colors = [GREY, BLUE, ORANGE]

    for ax, (metric, (title, unit)) in zip(axes, METRICS.items()):
        noisy_key = NOISY_KEYS[metric]
        entries = [
            (v13["metrics"][noisy_key], v13_path),
            (v13["metrics"][metric], v13_path),
            (v14["metrics"][metric], v14_path),
        ]
        values = [entry[0]["mean"] for entry in entries]
        errors = np.array(
            [
                [
                    value - entry[0]["hierarchical_ci95"][0]
                    for value, entry in zip(values, entries)
                ],
                [
                    entry[0]["hierarchical_ci95"][1] - value
                    for value, entry in zip(values, entries)
                ],
            ]
        )
        bars = ax.bar(
            models,
            values,
            yerr=errors,
            capsize=4,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}" for v in values],
            padding=3,
            fontsize=8,
        )
        spread = max(values) - min(values)
        margin = max(spread * 0.38, 0.025 if unit == "Score" else 0.2)
        ax.set_ylim(min(values) - margin, max(values) + margin * 1.5)
        for model, value, (_, source) in zip(models, values, entries):
            record(
                figure_name,
                "final",
                dataset_label,
                model,
                metric,
                value,
                source,
                "Three-seed hierarchical mean; bars show 95% CI.",
            )

    fig.suptitle(
        f"Final model performance — {dataset_label}\n"
        "Means across three seeds; error bars are hierarchical 95% confidence intervals",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def plot_paired_deltas(output: Path) -> None:
    datasets = [
        ("standard", "VoiceBank-DEMAND"),
        ("external", "DNS1 external"),
    ]
    figure_name = "02_v14_2_paired_deltas"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.ravel()

    for ax, (metric, (title, unit)) in zip(axes, METRICS.items()):
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        labels: list[str] = []
        for dataset, label in datasets:
            _, _, path = final_paths(dataset)
            value = read_json(path)["metrics"][metric]
            means.append(value["mean_delta"])
            lower.append(value["mean_delta"] - value["hierarchical_ci95"][0])
            upper.append(value["hierarchical_ci95"][1] - value["mean_delta"])
            labels.append(label)
            record(
                figure_name,
                "final",
                label,
                "V14.2 − V13",
                metric,
                value["mean_delta"],
                path,
                f"Paired hierarchical 95% CI: {value['hierarchical_ci95']}",
            )
        positions = np.arange(len(labels))
        ax.errorbar(
            means,
            positions,
            xerr=np.array([lower, upper]),
            fmt="o",
            markersize=8,
            color=ORANGE,
            ecolor="#333333",
            capsize=5,
            linewidth=1.5,
        )
        ax.axvline(0, color="#222222", linewidth=1, linestyle="--")
        ax.set_yticks(positions, labels)
        ax.set_xlabel(f"V14.2 − V13 ({unit})")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        for x, y in zip(means, positions):
            ax.annotate(
                f"{x:+.4f}" if abs(x) < 1 else f"{x:+.3f}",
                (x, y),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )

    fig.suptitle(
        "Paired effect of privileged distillation (V14.2 versus V13)\n"
        "Positive values favour V14.2; intervals crossing zero are inconclusive",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def plot_seed_robustness(output: Path) -> None:
    figure_name = "03_three_seed_robustness"
    datasets = [
        ("standard", "VoiceBank-DEMAND"),
        ("external", "DNS1 external"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.8), sharex=True)
    seeds = [1200, 1201, 1202]
    for row, (dataset, dataset_label) in enumerate(datasets):
        v13_path, v14_path, _ = final_paths(dataset)
        v13 = read_json(v13_path)["metrics"]
        v14 = read_json(v14_path)["metrics"]
        for col, (metric, (title, unit)) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for label, values, color, source in [
                ("V13", v13[metric]["seed_values"], BLUE, v13_path),
                ("V14.2", v14[metric]["seed_values"], ORANGE, v14_path),
            ]:
                ax.plot(
                    seeds,
                    values,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=label,
                )
                for seed, value in zip(seeds, values):
                    record(
                        figure_name,
                        "final",
                        dataset_label,
                        f"{label} seed {seed}",
                        metric,
                        value,
                        source,
                        "Per-seed aggregate.",
                    )
            ax.set_title(title)
            ax.set_ylabel(f"{dataset_label}\n{unit}" if col == 0 else unit)
            ax.set_xticks(seeds)
            ax.tick_params(axis="x", rotation=20)
            if row == 0 and col == 3:
                ax.legend(loc="best")
    fig.suptitle(
        "Reproducibility across the three frozen training seeds",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def plot_runtime(output: Path) -> None:
    figure_name = "04_streaming_efficiency"
    v8_path = "results/runtime/v8_native_streaming_7800x3d.json"
    structural_path = "results/runtime/v12_structural_latency_7800x3d.json"
    v13_path = "results/runtime/v13_final_gru_seed1200_cpu.json"
    v14_path = "results/runtime/v14_2_distilled_seed1200_cpu.json"
    v8 = read_json(v8_path)
    structural = read_json(structural_path)["variants"]
    v13 = read_json(v13_path)
    v14 = read_json(v14_path)
    rows = [
        (
            "V8 Mamba",
            v8["parameters"] / 1e6,
            v8["frame_time_ms"]["p95"],
            v8["streaming_rtf"],
            v8["state_tensor_mebibytes_fp32"],
            v8_path,
        ),
        (
            "V12 Mamba",
            structural["mamba_control"]["parameters"] / 1e6,
            structural["mamba_control"]["frame_time_ms"]["p95"],
            structural["mamba_control"]["streaming_rtf"],
            structural["mamba_control"]["state_tensor_mebibytes_fp32"],
            structural_path,
        ),
        (
            "V12 GRU",
            structural["gru_matched"]["parameters"] / 1e6,
            structural["gru_matched"]["frame_time_ms"]["p95"],
            structural["gru_matched"]["streaming_rtf"],
            structural["gru_matched"]["state_tensor_mebibytes_fp32"],
            structural_path,
        ),
        (
            "V13 GRU-T1",
            v13["parameters"] / 1e6,
            v13["frame_time_ms"]["p95"],
            v13["streaming_rtf"],
            v13["state_tensor_mebibytes_fp32"],
            v13_path,
        ),
        (
            "V14.2 final",
            v14["parameters"] / 1e6,
            v14["frame_time_ms"]["p95"],
            v14["streaming_rtf"],
            v14["state_tensor_mebibytes_fp32"],
            v14_path,
        ),
    ]
    labels = [row[0] for row in rows]
    colors = [PURPLE, PURPLE, GREEN, BLUE, ORANGE]
    panels = [
        (1, "Parameters", "Million"),
        (2, "Frame time (p95)", "ms"),
        (3, "Streaming real-time factor", "RTF"),
        (4, "Persistent state", "MiB (FP32)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, (index, title, unit) in zip(axes.ravel(), panels):
        values = [row[index] for row in rows]
        bars = ax.bar(labels, values, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)
        if index == 2:
            ax.axhline(
                10.0,
                color=RED,
                linestyle="--",
                linewidth=1.2,
                label="10 ms hop deadline",
            )
            ax.legend()
        if index == 3:
            ax.axhline(
                1.0,
                color=RED,
                linestyle="--",
                linewidth=1.2,
                label="Real-time boundary",
            )
            ax.legend()
        for row, value in zip(rows, values):
            record(
                figure_name,
                "deployment",
                "AMD Ryzen 7 7800X3D, one CPU thread",
                row[0],
                title,
                value,
                row[5],
                "All models use 20 ms algorithmic latency; V12 entries are structural benchmarks.",
            )
    fig.suptitle(
        "Causal streaming efficiency and memory footprint\n"
        "Lower is better in every panel except that parameter count is descriptive",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def plot_preservation_development(output: Path) -> None:
    figure_name = "05_preservation_models_dev400"
    rows = [
        (
            "V14.2",
            "results/v14/distillation/winner_dev400/summary.json",
        ),
        (
            "V15 quiet",
            "results/v15/preservation/quiet_level_seed1200/"
            "voicebank_dev400/summary.json",
        ),
        (
            "V15 identity",
            "results/v15/preservation/quiet_level_identity_seed1200/"
            "voicebank_dev400/summary.json",
        ),
        (
            "V15 causal",
            "results/v15/preservation/causal_preservation_gate_seed1200/"
            "voicebank_dev400/summary.json",
        ),
        (
            "V16 oracle",
            "results/v16/oracle_gate_seed16040/voicebank_dev400/summary.json",
        ),
    ]
    metric_keys = list(METRICS)
    loaded = [(name, path, read_json(path)["metrics"]) for name, path in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    colors = [ORANGE, GREEN, "#5CB8B2", BLUE, PURPLE]
    for ax, metric in zip(axes.ravel(), metric_keys):
        title, unit = METRICS[metric]
        values = [item[2][metric] for item in loaded]
        baseline = values[0]
        deltas = [value - baseline for value in values]
        bars = ax.bar(
            [item[0] for item in loaded],
            deltas,
            color=colors,
            edgecolor="white",
        )
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel(f"Change from V14.2 ({unit})")
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(
            bars,
            labels=[f"{value:+.4f}" for value in deltas],
            padding=2,
            fontsize=7.5,
        )
        for name, path, metrics in loaded:
            record(
                figure_name,
                "development",
                "VoiceBank locked development subset (400 files)",
                name,
                metric,
                metrics[metric],
                path,
                f"Figure shows delta from V14.2 value {baseline:.8f}; single seed.",
            )
    fig.suptitle(
        "Preservation-model development study (V15–V16)\n"
        "Single-seed locked development subset; values are changes from V14.2, not final-test claims",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def controller_candidates() -> list[tuple[str, dict[str, float], str]]:
    candidates: list[tuple[str, dict[str, float], str]] = []
    for label, path in [
        ("R5 utility", "results/v17/recipe5_selection/summary.json"),
        ("R5b balanced", "results/v17/recipe5b_selection/summary.json"),
        ("R6 statistics", "results/v17/recipe6_selection/summary.json"),
    ]:
        metrics = read_json(path)["selected_summary"]["metrics"]
        candidates.append((label, metrics, path))
    r6b_path = "results/v17/recipe6b_policy_selection/summary.json"
    candidates.append(
        (
            "R6b policy",
            read_json(r6b_path)["result"]["metrics"],
            r6b_path,
        )
    )
    r7_path = "results/v17/recipe7_conclusion.json"
    r7 = read_json(r7_path)["candidates"]
    for label, key in [
        ("R7a burn-in", "recipe7a_burn_in"),
        ("R7b prefix", "recipe7b_explicit_prefix"),
    ]:
        candidates.append((label, r7[key], r7_path))
    r8_path = "results/v18/recipe8_conclusion.json"
    candidates.append(("R8 two-stage", read_json(r8_path)["recipe8"], r8_path))
    return candidates


def plot_controller_study(output: Path) -> None:
    figure_name = "06_controller_recipe_outcomes"
    candidates = controller_candidates()
    panels = [
        (
            "avoidable_violation_rate",
            "Avoidable violation rate",
            0.20,
            "maximum",
        ),
        (
            "mean_constraint_violation",
            "Mean constraint violation",
            1.50,
            "maximum",
        ),
        ("mean_utility", "Mean utility", 2.50, "minimum"),
        ("mean_pesq_delta", "Mean PESQ improvement", 0.33, "minimum"),
        (
            "si_sdr_violation_rate",
            "SI-SDR violation rate",
            0.10,
            "maximum",
        ),
        ("stoi_violation_rate", "STOI violation rate", 0.18, "maximum"),
        ("estoi_violation_rate", "ESTOI violation rate", 0.11, "maximum"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14.0))
    flat_axes = axes.ravel()
    labels = [item[0] for item in candidates]
    colors = [RED, "#D97941", GREEN, "#4FA3A5", BLUE, PURPLE, ORANGE]
    for ax, (metric, title, threshold, direction) in zip(flat_axes, panels):
        values = [float(item[1][metric]) for item in candidates]
        bars = ax.bar(labels, values, color=colors, edgecolor="white")
        ax.axhline(
            threshold,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
        )
        ax.set_title(f"{title} (frozen {direction} {threshold:g})")
        ax.tick_params(axis="x", rotation=28)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        for (label, _, source), value in zip(candidates, values):
            record(
                figure_name,
                "training-domain calibration",
                "Controller calibration set (638 items)",
                label,
                metric,
                value,
                source,
                "Development-only frozen recipe selection; not external or standard test.",
            )
    flat_axes[-1].axis("off")
    flat_axes[-1].text(
        0.03,
        0.88,
        "Interpretation",
        fontsize=12,
        fontweight="bold",
        transform=flat_axes[-1].transAxes,
    )
    flat_axes[-1].text(
        0.03,
        0.72,
        "• Lower is better for violation metrics.\n"
        "• Higher is better for utility and PESQ improvement.\n"
        "• No recipe crossed the frozen 0.20 avoidable-violation gate.\n"
        "• R7a was the best balanced controller; R8 was not promoted.",
        fontsize=10,
        linespacing=1.6,
        va="top",
        transform=flat_axes[-1].transAxes,
    )
    fig.suptitle(
        "Preservation-controller recipe study (V17–V18)\n"
        "Training-domain calibration only; dashed lines are frozen acceptance thresholds",
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def clean_legacy_name(name: str) -> str:
    name = re.sub(r"^cnvqg_", "", name)
    name = re.sub(r"_test_full$", "", name)
    replacements = {
        "noise_adaptive_tf_mamba_": "NA-TF ",
        "continuous_adaptive_tf_mamba_": "CA-TF ",
        "streaming_hybrid_": "Streaming hybrid ",
        "hybrid_tf_": "Hybrid TF ",
        "mamba_": "Mamba ",
        "residual_mamba_": "Residual Mamba ",
    }
    for prefix, replacement in replacements.items():
        if name.startswith(prefix):
            name = replacement + name[len(prefix) :]
            break
    return name.replace("_", " ")


def plot_legacy_test_overview(output: Path) -> None:
    figure_name = "07_historical_test_exploration"
    summaries = sorted(
        (ROOT / "results" / "metrics").glob("*/summary.json")
    )
    rows: list[tuple[str, str, dict[str, float]]] = []
    for path in summaries:
        if not path.parent.name.endswith("_test_full"):
            continue
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
        if not all(metric in metrics for metric in METRICS):
            continue
        rows.append(
            (
                clean_legacy_name(path.parent.name),
                str(path.relative_to(ROOT)),
                metrics,
            )
        )
    rows.sort(key=lambda item: item[2]["enhanced_pesq"])
    labels = [row[0] for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 4, figsize=(17.0, max(9.0, len(rows) * 0.38)))
    for index, (ax, (metric, (title, unit))) in enumerate(
        zip(axes, METRICS.items())
    ):
        values = [row[2][metric] for row in rows]
        noisy_values = [row[2][NOISY_KEYS[metric]] for row in rows]
        colors = [
            plt.cm.viridis(0.15 + 0.75 * rank / max(len(rows) - 1, 1))
            for rank in range(len(rows))
        ]
        ax.barh(positions, values, color=colors, edgecolor="white")
        ax.axvline(
            float(np.median(noisy_values)),
            color=RED,
            linestyle="--",
            linewidth=1.1,
            label="Noisy input",
        )
        ax.set_title(title)
        ax.set_xlabel(unit)
        if index == 0:
            ax.set_yticks(positions, labels, fontsize=7.5)
        else:
            ax.set_yticks(positions, [])
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=7)
        for (label, source, _), value in zip(rows, values):
            record(
                figure_name,
                "historical exploratory",
                "VoiceBank-DEMAND test (reused during early exploration)",
                label,
                metric,
                value,
                source,
                "Appendix/context only; repeated test evaluation invalidates model-selection ranking.",
            )
    fig.suptitle(
        "Historical model evaluations across all recorded objective metrics\n"
        "Exploratory appendix only — the test set was repeatedly evaluated and this is not a valid selection leaderboard",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, output, figure_name)


def aggregate_seed_curves(
    paths: list[str],
    columns: list[tuple[str, str]],
    title: str,
    figure_name: str,
    output: Path,
    evidence_tier: str,
    frozen_epoch: int | None = None,
) -> None:
    frames = []
    for path in paths:
        frame = pd.read_csv(ROOT / path)
        frame["source"] = path
        frames.append(frame)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    for ax, (column, label) in zip(axes.ravel(), columns):
        available = [frame for frame in frames if column in frame.columns]
        if not available:
            ax.axis("off")
            continue
        for seed_index, frame in enumerate(available):
            ax.plot(
                frame["epoch"],
                frame[column],
                marker="o",
                alpha=0.42,
                linewidth=1,
                color=[BLUE, ORANGE, GREEN][seed_index % 3],
                label=f"Seed {1200 + seed_index}",
            )
            for epoch, value in zip(frame["epoch"], frame[column]):
                record(
                    figure_name,
                    evidence_tier,
                    "Training/locked validation",
                    f"Seed {1200 + seed_index}",
                    column,
                    float(value),
                    paths[seed_index],
                    f"Epoch {int(epoch)}",
                )
        common_epochs = sorted(
            set.intersection(*[set(frame["epoch"]) for frame in available])
        )
        if common_epochs:
            matrix = np.array(
                [
                    [
                        float(
                            frame.loc[frame["epoch"] == epoch, column].iloc[0]
                        )
                        for epoch in common_epochs
                    ]
                    for frame in available
                ]
            )
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0, ddof=1) if len(available) > 1 else np.zeros_like(mean)
            ax.plot(
                common_epochs,
                mean,
                color="#111111",
                linewidth=2.3,
                label="Mean",
                zorder=5,
            )
            ax.fill_between(
                common_epochs,
                mean - std,
                mean + std,
                color="#111111",
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if frozen_epoch is not None:
            ax.axvline(
                frozen_epoch,
                color=ORANGE,
                linestyle=":",
                linewidth=1.4,
                alpha=0.9,
            )
    axes.ravel()[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def plot_final_training_curves(output: Path) -> None:
    v13_paths = [
        f"checkpoints/v12/full/gru_matched_time1_seed{seed}/metrics.csv"
        for seed in [1200, 1201, 1202]
    ]
    aggregate_seed_curves(
        v13_paths,
        [
            ("train_loss_total", "Training total loss"),
            ("val_loss_total", "Validation total loss"),
            ("val_loss_noise_prediction", "Validation noise loss"),
            ("val_loss_magnitude", "Validation magnitude loss"),
            ("val_perceptual_enhanced_pesq", "Validation PESQ"),
            (
                "val_perceptual_si_sdr_improvement",
                "Validation SI-SDR improvement",
            ),
        ],
        "V13 backbone learning curves across three frozen seeds",
        "08_v13_training_curves",
        output,
        "training",
    )
    v14_paths = [
        "checkpoints/v14/distillation/mag_005/metrics.csv",
        "checkpoints/v14/distillation/mag_005_seed1201_fixed_epoch3/metrics.csv",
        "checkpoints/v14/distillation/mag_005_seed1202_fixed_epoch3/metrics.csv",
    ]
    aggregate_seed_curves(
        v14_paths,
        [
            ("train_total", "Training total loss"),
            ("val_total", "Validation total loss"),
            ("val_supervised", "Validation supervised loss"),
            ("val_distilled_total", "Validation distillation loss"),
            ("val_compressed_complex", "Validation complex loss"),
            ("val_log_magnitude", "Validation log-magnitude loss"),
        ],
        "V14.2 privileged-distillation curves across three frozen seeds\n"
        "The frozen deployment checkpoint is epoch 3",
        "09_v14_2_distillation_curves",
        output,
        "training",
        frozen_epoch=3,
    )


def plot_controller_training_curves(output: Path) -> None:
    figure_name = "10_controller_training_curves"
    paths = [
        (
            "R5",
            "checkpoints/v17/utility_safety_recipe5_seed17044/metrics.csv",
        ),
        (
            "R5b",
            "checkpoints/v17/utility_safety_recipe5b_balanced_seed17045/metrics.csv",
        ),
        (
            "R6",
            "checkpoints/v17/utility_safety_recipe6_statistics_seed17046/metrics.csv",
        ),
        (
            "R7a",
            "checkpoints/v17/utility_safety_recipe7a_burnin_seed17047/metrics.csv",
        ),
        (
            "R7b",
            "checkpoints/v17/utility_safety_recipe7b_prefix_seed17048/metrics.csv",
        ),
        (
            "R8",
            "checkpoints/v18/utility_safety_recipe8_two_stage_seed18000/metrics.csv",
        ),
    ]
    panels = [
        ("val_loss_total", "Validation total loss"),
        ("val_loss_gate_utility", "Utility loss"),
        ("val_loss_gate_violation", "Violation loss"),
        ("val_loss_gate_feasibility", "Feasibility loss"),
        ("val_loss_gate_policy", "Policy loss"),
        ("val_gate_strength_mean", "Mean predicted strength"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    palette = [RED, "#D97941", GREEN, "#4FA3A5", BLUE, PURPLE]
    for ax, (column, title) in zip(axes.ravel(), panels):
        plotted = 0
        for (label, path), color in zip(paths, palette):
            source = ROOT / path
            if not source.is_file():
                continue
            frame = pd.read_csv(source)
            if column not in frame.columns:
                continue
            values = frame[column].replace([np.inf, -np.inf], np.nan)
            if not values.notna().any():
                continue
            ax.plot(
                frame["epoch"],
                values,
                marker="o",
                linewidth=1.8,
                color=color,
                label=label,
            )
            plotted += 1
            for epoch, value in zip(frame["epoch"], values):
                if pd.notna(value):
                    record(
                        figure_name,
                        "training-domain calibration",
                        "Controller validation",
                        label,
                        column,
                        float(value),
                        path,
                        f"Epoch {int(epoch)}",
                    )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 2, 4, 6, 8])
        if plotted == 0:
            ax.text(
                0.5,
                0.5,
                "Not logged by these recipes",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=GREY,
            )
    axes.ravel()[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "V17–V18 controller optimisation curves\n"
        "Loss magnitudes are comparable only where the recipe retained the same objective definition",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output, figure_name)


def write_outputs(output: Path) -> None:
    provenance_path = output / "figure_provenance.csv"
    with provenance_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure",
                "evidence_tier",
                "dataset",
                "model",
                "metric",
                "value",
                "source",
                "note",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(PROVENANCE)

    stems = sorted(path.stem for path in output.glob("*.png"))
    lines = [
        "# Dissertation figure set",
        "",
        "Generated by `scripts/generate_thesis_figures.py` from the frozen local "
        "results.",
        "",
        "## Evidence policy",
        "",
        "- Figures 01–04 contain the defensible final/deployment evidence.",
        "- Figures 05–06 contain development-only preservation studies.",
        "- Figure 07 is historical context only; it is not a valid model-selection "
        "leaderboard because the standard test set was repeatedly evaluated.",
        "- Figures 08–10 are optimisation diagnostics, not independent quality "
        "claims.",
        "- Every plotted value and its source file is recorded in "
        "`figure_provenance.csv`.",
        "",
        "## Figures",
        "",
    ]
    descriptions = {
        "01_final_standard_absolute": "Final three-seed VoiceBank-DEMAND metrics.",
        "01_final_external_absolute": "Final three-seed DNS1 external metrics.",
        "02_v14_2_paired_deltas": "Paired V14.2 minus V13 effects with 95% CIs.",
        "03_three_seed_robustness": "Per-seed reproducibility on both final sets.",
        "04_streaming_efficiency": "Parameters, latency, RTF, and recurrent state.",
        "05_preservation_models_dev400": "V15–V16 locked-development comparison.",
        "06_controller_recipe_outcomes": "V17–V18 controller outcomes and gates.",
        "07_historical_test_exploration": "Historical exploratory model overview.",
        "08_v13_training_curves": "V13 backbone training and validation curves.",
        "09_v14_2_distillation_curves": "V14.2 three-seed distillation curves.",
        "10_controller_training_curves": "V17–V18 controller loss curves.",
    }
    for stem in stems:
        lines.append(f"- `{stem}.png` / `{stem}.pdf`: {descriptions.get(stem, '')}")
    lines.extend(
        [
            "",
            f"Provenance rows: {len(PROVENANCE)}",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for PNG, PDF, README, and provenance CSV outputs.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    plot_final_absolute(output, "standard")
    plot_final_absolute(output, "external")
    plot_paired_deltas(output)
    plot_seed_robustness(output)
    plot_runtime(output)
    plot_preservation_development(output)
    plot_controller_study(output)
    plot_legacy_test_overview(output)
    plot_final_training_curves(output)
    plot_controller_training_curves(output)
    write_outputs(output)

    print(f"Wrote {len(list(output.glob('*.png')))} PNG figures")
    print(f"Wrote {len(list(output.glob('*.pdf')))} PDF figures")
    print(f"Wrote {len(PROVENANCE)} provenance rows")
    print(output)


if __name__ == "__main__":
    main()
