#!/usr/bin/env python3
"""Train zero-initialised explicit-scale decoder adapters on the frozen V8 model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v8/train_v8_direct_scalar_full.yaml"
INITIAL_CHECKPOINT = "checkpoints/v8/v8_direct_scalar_full_scratch/best.pt"
REFERENCE_RESULTS = ROOT / "results/v8/v8_direct_scalar_full_scratch/locked400_full"
SEARCH_METADATA = "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
LOCKED_METADATA = "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"

MIN_PESQ_GAIN = 0.02
MAX_SI_SDR_LOSS = 0.10
MAX_STOI_LOSS = 0.001
MAX_ESTOI_LOSS = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v11_decoder_main")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


class Programme:
    variants = {
        "scale_context": False,
        "scale_frequency_context": True,
    }

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = "".join(
            c if c.isalnum() or c in "_.-" else "-" for c in args.run_id
        )
        self.config_dir = ROOT / "configs/v11/generated" / self.run_id
        self.checkpoint_dir = ROOT / "checkpoints/v11" / self.run_id
        self.result_dir = ROOT / "results/v11" / self.run_id
        self.report_path = self.result_dir / "decision_report.json"
        self.epochs = int(os.environ.get("V11_DECODER_EPOCHS", "4"))
        self.batches = int(os.environ.get("V11_DECODER_BATCHES", "600"))
        self.base = yaml.safe_load(BASE_CONFIG.read_text())
        self.report: dict[str, Any] = {
            "run_id": self.run_id,
            "initial_checkpoint": INITIAL_CHECKPOINT,
            "reference": str(REFERENCE_RESULTS.relative_to(ROOT)),
            "test_set_used": False,
            "frozen_backbone": True,
            "thresholds": {
                "min_pesq_gain": MIN_PESQ_GAIN,
                "max_si_sdr_loss": MAX_SI_SDR_LOSS,
                "max_stoi_loss": MAX_STOI_LOSS,
                "max_estoi_loss": MAX_ESTOI_LOSS,
            },
            "variants": {},
        }

    def save(self) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(self.report, indent=2) + "\n")

    def config(self, name: str, use_frequency_coordinate: bool) -> dict[str, Any]:
        config = deepcopy(self.base)
        config["project"]["seed"] = 11101
        config["data"].update(
            {
                "val_metadata": SEARCH_METADATA,
                "batch_size": 6,
                "num_workers": 8,
            }
        )
        config["model"].update(
            {
                "architecture": "causal_scale_aware_mamba_v11",
                "scale_adapter_hidden_channels": 16,
                "use_frequency_coordinate": use_frequency_coordinate,
                "scale_adapter_residual_scale": 0.1,
            }
        )
        config["training"].update(
            {
                "epochs": self.epochs,
                "learning_rate": 3e-4,
                "max_train_batches": self.batches,
                "init_checkpoint": INITIAL_CHECKPOINT,
                "init_strict": False,
                "trainable_parameter_patterns": ["decoder.scale_adapter"],
                "early_stopping": {"enabled": False},
                "lr_scheduler": {"name": "none"},
                "perceptual_validation": {"enabled": True, "max_items": 100},
                "save_every_epoch": False,
            }
        )
        config["paths"] = {
            "checkpoint_dir": str(self.checkpoint_dir.relative_to(ROOT)),
            "experiment_name": name,
        }
        return config

    def prepare(self) -> dict[str, Path]:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, coordinate in self.variants.items():
            path = self.config_dir / f"{name}.yaml"
            path.write_text(yaml.safe_dump(self.config(name, coordinate), sort_keys=False))
            paths[name] = path
        self.report["prepared_configs"] = [
            str(path.relative_to(ROOT)) for path in paths.values()
        ]
        self.save()
        return paths

    def run(self, *arguments: str) -> None:
        command = [sys.executable, *arguments]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    def train(self, name: str, config: Path) -> Path:
        checkpoint = self.checkpoint_dir / name / "best.pt"
        if not checkpoint.exists():
            self.run(
                "scripts/train.py",
                "--config",
                str(config.relative_to(ROOT)),
                "--device",
                self.args.device,
            )
        return checkpoint

    def evaluate(self, name: str, checkpoint: Path) -> dict[str, float]:
        output = self.result_dir / name / "locked400"
        summary = output / "summary.json"
        if not summary.exists():
            self.run(
                "scripts/evaluate.py",
                "--checkpoint",
                str(checkpoint.relative_to(ROOT)),
                "--metadata",
                LOCKED_METADATA,
                "--output-dir",
                str(output.relative_to(ROOT)),
                "--device",
                self.args.device,
            )
        return json.loads(summary.read_text())["metrics"]

    def compare(self) -> dict[str, Any]:
        output = self.result_dir / "comparison"
        comparison = output / "comparison.json"
        if not comparison.exists():
            arguments = [
                "scripts/compare_evaluations.py",
                "--reference",
                f"v8={REFERENCE_RESULTS}",
            ]
            for name in self.variants:
                arguments.extend(
                    ("--candidate", f"{name}={self.result_dir / name / 'locked400'}")
                )
            arguments.extend(
                ("--output-dir", str(output), "--bootstrap-samples", "20000")
            )
            self.run(*arguments)
        return json.loads(comparison.read_text())

    @staticmethod
    def metric_gate(candidate: dict[str, float], reference: dict[str, float]) -> bool:
        return (
            candidate["enhanced_pesq"] >= reference["enhanced_pesq"] + MIN_PESQ_GAIN
            and candidate["enhanced_si_sdr"] >= reference["enhanced_si_sdr"] - MAX_SI_SDR_LOSS
            and candidate["enhanced_stoi"] >= reference["enhanced_stoi"] - MAX_STOI_LOSS
            and candidate["enhanced_estoi"] >= reference["enhanced_estoi"] - MAX_ESTOI_LOSS
        )

    def main(self) -> None:
        paths = self.prepare()
        if self.args.prepare_only:
            print(json.dumps(self.report, indent=2))
            return
        metrics = {}
        for name, path in paths.items():
            metrics[name] = self.evaluate(name, self.train(name, path))
            self.report["variants"][name] = metrics[name]
            self.save()
        reference = json.loads((REFERENCE_RESULTS / "summary.json").read_text())["metrics"]
        comparison = self.compare()
        for name in self.variants:
            low = comparison["paired_bootstrap"][name]["enhanced_pesq"]["ci95"][0]
            self.report["variants"][name]["promoted"] = (
                self.metric_gate(metrics[name], reference) and float(low) > 0.0
            )
        winners = [
            name for name in self.variants
            if self.report["variants"][name]["promoted"]
        ]
        winner = max(winners, key=lambda n: metrics[n]["enhanced_pesq"]) if winners else None
        self.report["winner"] = winner
        self.report["decision"] = (
            f"promote_{winner}" if winner else "stop_decoder_adapter_no_promotion"
        )
        self.report["phase_deferred"] = winner is None
        self.save()
        print(json.dumps(self.report, indent=2))


if __name__ == "__main__":
    Programme(parse_args()).main()
