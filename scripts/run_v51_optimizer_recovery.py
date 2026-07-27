#!/usr/bin/env python3
"""Recover V5.1 training stability with guarded optimizer/loss ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/v5/generated/v51_main/arch_v51_log_256.yaml")
BASE_CHECKPOINT = Path(
    "checkpoints/v51_tournament/v51_main/arch_v51_log_256/best.pt"
)
BASE_METRICS = Path(
    "checkpoints/v51_tournament/v51_main/arch_v51_log_256/metrics.csv"
)
METRIC_NAMES = (
    "val_perceptual_enhanced_pesq",
    "val_perceptual_enhanced_si_sdr",
    "val_perceptual_enhanced_stoi",
    "val_perceptual_enhanced_estoi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v51_optimizer_recovery")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def metrics_from_row(row: dict[str, str]) -> dict[str, float]:
    return {name: float(row[name]) for name in METRIC_NAMES}


def best_baseline() -> dict[str, float]:
    rows = read_rows(ROOT / BASE_METRICS)
    row = max(rows, key=lambda item: float(item[METRIC_NAMES[0]]))
    return metrics_from_row(row)


def passes_no_harm(
    candidate: dict[str, float], reference: dict[str, float], pesq_slack: float = 0.003
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "pesq_stable": candidate[METRIC_NAMES[0]] >= reference[METRIC_NAMES[0]] - pesq_slack,
        "si_sdr_no_harm": candidate[METRIC_NAMES[1]] >= reference[METRIC_NAMES[1]] - 0.15,
        "stoi_no_harm": candidate[METRIC_NAMES[2]] >= reference[METRIC_NAMES[2]] - 0.002,
        "estoi_no_harm": candidate[METRIC_NAMES[3]] >= reference[METRIC_NAMES[3]] - 0.003,
    }
    return all(gates.values()), gates


class Recovery:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = ROOT / "results" / "v51_optimizer_recovery" / args.run_id
        self.config_dir = ROOT / "configs" / "v5" / "generated" / args.run_id
        self.checkpoint_root = ROOT / "checkpoints" / "v51_optimizer_recovery" / args.run_id
        self.log_dir = self.run_dir / "logs"
        for path in (self.run_dir, self.config_dir, self.checkpoint_root, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.baseline = best_baseline()
        self.report: dict[str, Any] = {
            "run_id": args.run_id,
            "source_checkpoint": str(BASE_CHECKPOINT),
            "baseline": self.baseline,
            "trials": {},
            "status": "prepared",
        }

    def save(self) -> None:
        (self.run_dir / "decision_report.json").write_text(
            json.dumps(self.report, indent=2) + "\n"
        )

    def config(self, name: str, optimizer: str, muon_lr: float, rescue: bool) -> Path:
        config = yaml.safe_load((ROOT / BASE_CONFIG).read_text())
        config["project"]["seed"] = 1210
        config["paths"]["checkpoint_dir"] = str(self.checkpoint_root)
        config["paths"]["experiment_name"] = name
        training = config["training"]
        training["epochs"] = 4
        training["init_checkpoint"] = str(ROOT / BASE_CHECKPOINT)
        training["init_strict"] = True
        training["early_stopping"] = {"enabled": False}
        training["lr_scheduler"] = {"name": "none"}
        training["optimizer"] = (
            {
                "name": "muon",
                "muon_learning_rate": muon_lr,
                "adamw_learning_rate": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "muon_weight_decay": 0.00001,
            }
            if optimizer == "muon"
            else {"name": "adamw"}
        )
        training["learning_rate"] = 0.0001
        if rescue:
            loss = config["loss"]
            loss["si_sdr_weight"] = 0.005
            loss["stft_weight"] = 0.15
            loss["phase_weight"] = 0.25
            loss["group_delay_weight"] = 0.10
            loss["instantaneous_frequency_weight"] = 0.10
        path = self.config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def command(self, tag: str, arguments: list[str]) -> bool:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START {tag}", flush=True)
        with (self.log_dir / f"{tag}.log").open("a") as log:
            process = subprocess.Popen(
                arguments,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            status = process.wait()
        print(f"[{datetime.now().isoformat(timespec='seconds')}] END {tag}: {status}", flush=True)
        return status == 0

    def train(self, name: str, config: Path, epoch: int, resume: bool) -> bool:
        checkpoint_dir = self.checkpoint_root / name
        metrics_path = checkpoint_dir / "metrics.csv"
        if metrics_path.exists():
            completed = {int(row["epoch"]) for row in read_rows(metrics_path)}
            if epoch in completed:
                print(f"SKIP {name} epoch {epoch}: already complete", flush=True)
                return True
        command = [
            sys.executable,
            "scripts/train.py",
            "--config", str(config),
            "--device", self.args.device,
            "--epochs", str(epoch),
            "--max-train-batches", "250",
            "--max-val-batches", "50",
        ]
        if resume:
            latest = checkpoint_dir / "latest.pt"
            if not latest.exists():
                return False
            command += ["--resume", str(latest)]
        return self.command(f"{name}_epoch_{epoch}", command)

    def latest_metrics(self, name: str) -> dict[str, float]:
        rows = read_rows(self.checkpoint_root / name / "metrics.csv")
        return metrics_from_row(rows[-1])

    def initial_trials(self, specifications: list[tuple[str, str, float, bool]]) -> list[str]:
        survivors = []
        for name, optimizer, muon_lr, rescue in specifications:
            path = self.config(name, optimizer, muon_lr, rescue)
            if not self.train(name, path, 1, False):
                self.report["trials"][name] = {"error": "training_failed"}
                self.save()
                continue
            metrics = self.latest_metrics(name)
            passed, gates = passes_no_harm(metrics, self.baseline)
            self.report["trials"][name] = {
                "optimizer": optimizer,
                "muon_learning_rate": muon_lr if optimizer == "muon" else None,
                "loss_rescue": rescue,
                "epochs": {"1": metrics},
                "gates": {"1": gates},
            }
            if passed:
                survivors.append(name)
            self.save()
        return survivors

    def run(self) -> None:
        primary = [
            ("muon_3e4", "muon", 0.0003, False),
            ("muon_1e4", "muon", 0.0001, False),
            ("adamw_1e4", "adamw", 0.0, False),
        ]
        rescue = [
            ("muon_3e4_loss_rescue", "muon", 0.0003, True),
            ("adamw_1e4_loss_rescue", "adamw", 0.0, True),
        ]
        for specification in primary + rescue:
            self.config(*specification)
        self.save()
        if self.args.prepare_only:
            return

        survivors = self.initial_trials(primary)
        if not survivors:
            self.report["optimizer_only_outcome"] = "no_survivor; running loss rescue"
            self.save()
            survivors = self.initial_trials(rescue)
        if not survivors:
            self.report["status"] = "stopped_no_stable_candidate"
            self.save()
            print("No candidate passed the 250-batch no-harm gates.")
            return

        winner = max(
            survivors,
            key=lambda name: self.report["trials"][name]["epochs"]["1"][METRIC_NAMES[0]],
        )
        config = self.config_dir / f"{winner}.yaml"
        previous = self.report["trials"][winner]["epochs"]["1"]
        for epoch in (2, 3, 4):
            if not self.train(winner, config, epoch, True):
                self.report["status"] = f"stopped_training_failure_epoch_{epoch}"
                self.save()
                return
            metrics = self.latest_metrics(winner)
            passed, gates = passes_no_harm(metrics, previous)
            baseline_passed, baseline_gates = passes_no_harm(metrics, self.baseline)
            self.report["trials"][winner]["epochs"][str(epoch)] = metrics
            self.report["trials"][winner]["gates"][str(epoch)] = {
                **gates,
                **{f"baseline_{key}": value for key, value in baseline_gates.items()},
            }
            self.save()
            if not passed or not baseline_passed:
                self.report["status"] = f"stopped_regression_epoch_{epoch}"
                self.report["winner"] = winner
                self.save()
                print(f"Stopped {winner}: regression detected at epoch {epoch}.")
                return
            previous = metrics

        final_pesq = previous[METRIC_NAMES[0]]
        self.report["winner"] = winner
        self.report["status"] = (
            "passed_1000_batch_recovery"
            if final_pesq >= self.baseline[METRIC_NAMES[0]] + 0.01
            else "stable_but_no_material_pesq_gain"
        )
        self.save()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    Recovery(args).run()


if __name__ == "__main__":
    main()
