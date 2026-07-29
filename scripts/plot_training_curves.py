from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_log(log_path: Path) -> pd.DataFrame:
    rows = []
    current_row = {}

    epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")

    for line in log_path.read_text(errors="ignore").splitlines():
        epoch_match = epoch_pattern.search(line)

        if epoch_match:
            if current_row:
                rows.append(current_row)
                current_row = {}

            current_row["epoch"] = int(epoch_match.group(1))
            continue

        if line.startswith("Train: "):
            values = ast.literal_eval(line.replace("Train: ", "", 1))
            for key, value in values.items():
                current_row[f"train_{key}"] = value

        if line.startswith("Val: "):
            values = ast.literal_eval(line.replace("Val: ", "", 1))
            for key, value in values.items():
                current_row[f"val_{key}"] = value

    if current_row:
        rows.append(current_row)

    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    train_col = f"train_{metric}"
    val_col = f"val_{metric}"

    if train_col not in df.columns and val_col not in df.columns:
        return

    plt.figure()

    if train_col in df.columns:
        plt.plot(df["epoch"], df[train_col], marker="o", label="Train")

    if val_col in df.columns:
        plt.plot(df["epoch"], df[val_col], marker="o", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = output_dir / f"{metric}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    print("Wrote", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_log(log_path)

    csv_path = output_dir / "training_history.csv"
    df.to_csv(csv_path, index=False)

    print("Wrote", csv_path)
    print(df)

    metrics = [
        "loss_total",
        "loss_waveform_l1",
        "loss_si_sdr",
        "loss_stft",
        "loss_vq",
        "loss_mel",
        "loss_complex_stft",
        "loss_metric_d",
        "loss_metric_g",
        "vq_perplexity",
    ]

    if "learning_rate" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["learning_rate"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("learning_rate")
        plt.grid(True)
        plt.tight_layout()
        out_path = output_dir / "learning_rate.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print("Wrote", out_path)

    for metric in metrics:
        plot_metric(df, metric, output_dir)


if __name__ == "__main__":
    main()
