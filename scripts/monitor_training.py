#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a CN-VQG training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--process-match", default="scripts/train.py")
    parser.add_argument("--stale-minutes", type=float, default=45.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def matching_processes(pattern: str) -> list[dict[str, object]]:
    matches = []
    for process in psutil.process_iter(("pid", "cmdline", "create_time")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if pattern in command and "monitor_training.py" not in command:
                matches.append(
                    {
                        "pid": process.info["pid"],
                        "command": command,
                        "age_seconds": time.time() - float(process.info["create_time"]),
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def read_last_metric(path: Path) -> dict[str, object] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    row = rows[-1]
    selected = {
        "epoch": int(row["epoch"]),
        "learning_rate": float(row["learning_rate"]),
    }
    for key in (
        "train_loss_total",
        "val_loss_total",
        "val_loss_si_sdr",
        "val_vq_perplexity",
        "val_vq_active_fraction",
        "val_vq_dead_fraction",
    ):
        if row.get(key) not in (None, ""):
            selected[key] = float(row[key])
    return selected


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.csv"
    latest_path = run_dir / "latest.pt"
    log_path = run_dir / "train_full.log"
    processes = matching_processes(args.process_match)
    newest_checkpoint_age = None
    if latest_path.exists():
        newest_checkpoint_age = time.time() - latest_path.stat().st_mtime
    log_tail = ""
    if log_path.exists():
        with log_path.open(errors="replace") as file:
            file.seek(max(0, log_path.stat().st_size - 16000))
            log_tail = file.read()

    problems = []
    if not processes:
        problems.append("training process is not running")
    if (
        newest_checkpoint_age is not None
        and newest_checkpoint_age > args.stale_minutes * 60.0
    ):
        problems.append("latest checkpoint is stale")
    if "Non-finite loss" in log_tail or "Traceback (most recent call last)" in log_tail:
        problems.append("training log contains a recent failure")

    report = {
        "healthy": not problems,
        "problems": problems,
        "run_dir": str(run_dir),
        "processes": processes,
        "last_metric": read_last_metric(metrics_path),
        "latest_checkpoint_age_seconds": newest_checkpoint_age,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("healthy:", report["healthy"])
        print("problems:", "; ".join(problems) if problems else "none")
        print("processes:", len(processes))
        print("last metric:", report["last_metric"])
        print("latest checkpoint age (s):", newest_checkpoint_age)
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
