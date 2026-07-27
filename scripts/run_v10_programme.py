#!/usr/bin/env python3
"""Run the guarded V10 global-frequency and phase ablation programme.

The programme never reads test metadata. It uses a matched short screen,
locked-400 promotion, then optionally launches three fresh full-training seeds.
"""

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
SEARCH_METADATA = "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
LOCKED_METADATA = "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"

GLOBAL_MIN_PESQ = 0.03
MAX_SI_SDR_LOSS = 0.10
MAX_STOI_LOSS = 0.001
MAX_ESTOI_LOSS = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v10_main")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--execute-full",
        action="store_true",
        help="Train three fresh full seeds after every promotion gate passes.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


class Programme:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = safe_run_id(args.run_id)
        self.config_dir = ROOT / "configs/v10/generated" / self.run_id
        self.checkpoint_dir = ROOT / "checkpoints/v10" / self.run_id
        self.result_dir = ROOT / "results/v10" / self.run_id
        self.status_path = self.result_dir / "decision_report.json"
        self.screen_epochs = int(os.environ.get("V10_SCREEN_EPOCHS", "4"))
        self.screen_batches = int(os.environ.get("V10_SCREEN_BATCHES", "600"))
        self.base = yaml.safe_load(BASE_CONFIG.read_text())
        self.report: dict[str, Any] = {
            "run_id": self.run_id,
            "test_set_used": False,
            "native_streaming_required_before_deployment": True,
            "thresholds": {
                "global_min_pesq": GLOBAL_MIN_PESQ,
                "max_si_sdr_loss": MAX_SI_SDR_LOSS,
                "max_stoi_loss": MAX_STOI_LOSS,
                "max_estoi_loss": MAX_ESTOI_LOSS,
            },
            "screen": {},
            "phase": {},
            "full": {"launched": False, "configs": []},
        }

    def save(self) -> None:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.report, indent=2) + "\n")

    def config(
        self,
        name: str,
        *,
        attention: bool,
        attention_dim: int = 40,
        seed: int = 10001,
        stage: str = "screen",
        init_checkpoint: Path | None = None,
    ) -> dict[str, Any]:
        config = deepcopy(self.base)
        config["project"]["seed"] = seed
        config["data"].update(
            {
                "val_metadata": SEARCH_METADATA if stage != "full" else LOCKED_METADATA,
                "batch_size": 4,
                "num_workers": 8,
            }
        )
        config["model"].update(
            {
                "architecture": "causal_global_frequency_mamba_v10",
                "use_global_frequency_attention": attention,
                "frequency_attention_dim": attention_dim,
                "frequency_attention_heads": 4,
                "frequency_attention_expansion": 2,
                "frequency_residual_scale": 0.1,
            }
        )
        training = config["training"]
        if stage == "screen":
            training.update(
                {
                    "epochs": self.screen_epochs,
                    "max_train_batches": self.screen_batches,
                    "early_stopping": {"enabled": False},
                    "lr_scheduler": {"name": "none"},
                    "perceptual_validation": {"enabled": True, "max_items": 100},
                    "save_every_epoch": False,
                }
            )
        else:
            training.pop("max_train_batches", None)
            training.pop("init_checkpoint", None)
            training.pop("init_strict", None)
            training["perceptual_validation"] = {"enabled": True, "max_items": 400}
        config["paths"] = {
            "checkpoint_dir": str(self.checkpoint_dir.relative_to(ROOT)),
            "experiment_name": name,
        }
        return config

    def write_config(self, name: str, config: dict[str, Any]) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        path = self.config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def prepare_screen(self) -> dict[str, Path]:
        configs = {
            "control": self.config("control", attention=False),
            "global_d32": self.config("global_d32", attention=True, attention_dim=32),
            "global_d40": self.config("global_d40", attention=True, attention_dim=40),
        }
        return {name: self.write_config(name, config) for name, config in configs.items()}

    def run(self, *arguments: str) -> None:
        command = [sys.executable, *arguments]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    def train(self, name: str, config: Path) -> Path:
        checkpoint = self.checkpoint_dir / name / "best.pt"
        if not checkpoint.exists():
            self.run(
                "scripts/train.py",
                "--config", str(config.relative_to(ROOT)),
                "--device", self.args.device,
            )
        if not checkpoint.exists():
            raise RuntimeError(f"Training did not create {checkpoint}")
        return checkpoint

    def evaluate(self, stage: str, name: str, checkpoint: Path) -> dict[str, float]:
        output = self.result_dir / stage / name / "locked400"
        summary = output / "summary.json"
        if not summary.exists():
            self.run(
                "scripts/evaluate.py",
                "--checkpoint", str(checkpoint.relative_to(ROOT)),
                "--metadata", LOCKED_METADATA,
                "--output-dir", str(output.relative_to(ROOT)),
                "--device", self.args.device,
            )
        return json.loads(summary.read_text())["metrics"]

    @staticmethod
    def promoted(candidate: dict[str, float], reference: dict[str, float], gain: float) -> bool:
        return (
            candidate["enhanced_pesq"] >= reference["enhanced_pesq"] + gain
            and candidate["enhanced_si_sdr"] >= reference["enhanced_si_sdr"] - MAX_SI_SDR_LOSS
            and candidate["enhanced_stoi"] >= reference["enhanced_stoi"] - MAX_STOI_LOSS
            and candidate["enhanced_estoi"] >= reference["enhanced_estoi"] - MAX_ESTOI_LOSS
        )

    def compare(self, stage: str, reference: str, candidates: list[str]) -> dict[str, Any]:
        output = self.result_dir / stage / "comparison"
        if (output / "comparison.json").exists():
            return json.loads((output / "comparison.json").read_text())
        else:
            arguments = [
                "scripts/compare_evaluations.py",
                "--reference",
                f"{reference}={self.result_dir / stage / reference / 'locked400'}",
            ]
            for name in candidates:
                arguments.extend(
                    ("--candidate", f"{name}={self.result_dir / stage / name / 'locked400'}")
                )
            arguments.extend(("--output-dir", str(output), "--bootstrap-samples", "20000"))
            self.run(*arguments)
            return json.loads((output / "comparison.json").read_text())

    @staticmethod
    def positive_pesq_interval(comparison: dict[str, Any], name: str) -> bool:
        low = comparison["paired_bootstrap"][name]["enhanced_pesq"]["ci95"][0]
        return float(low) > 0.0

    def structural_gate(self, config: Path) -> None:
        output = self.result_dir / "structural_gates.json"
        if output.exists():
            return
        self.run(
            "scripts/check_v51_candidate.py",
            "--config", str(config.relative_to(ROOT)),
            "--output", str(output.relative_to(ROOT)),
            "--device", self.args.device,
        )

    def run_screen(self, paths: dict[str, Path]) -> tuple[str | None, Path | None]:
        self.structural_gate(paths["global_d40"])
        metrics: dict[str, dict[str, float]] = {}
        checkpoints: dict[str, Path] = {}
        for name, path in paths.items():
            checkpoints[name] = self.train(name, path)
            metrics[name] = self.evaluate("screen", name, checkpoints[name])
            self.report["screen"][name] = metrics[name]
            self.save()
        reference = metrics["control"]
        comparison = self.compare("screen", "control", ["global_d32", "global_d40"])
        for name in ("global_d32", "global_d40"):
            self.report["screen"][name]["promoted"] = self.promoted(
                metrics[name], reference, GLOBAL_MIN_PESQ
            ) and self.positive_pesq_interval(comparison, name)
        winners = [
            name for name in ("global_d32", "global_d40")
            if self.report["screen"][name]["promoted"]
        ]
        winner = max(winners, key=lambda name: metrics[name]["enhanced_pesq"]) if winners else None
        self.report["screen"]["winner"] = winner
        self.save()
        return (winner, checkpoints[winner]) if winner else (None, None)

    def prepare_full(self, selection: dict[str, Any]) -> list[Path]:
        paths = []
        for index, seed in enumerate((10101, 10102, 10103), start=1):
            name = f"full_seed{index}"
            config = self.config(
                name,
                attention=True,
                attention_dim=int(selection["attention_dim"]),
                seed=seed,
                stage="full",
            )
            paths.append(self.write_config(name, config))
        self.report["full"]["selection"] = selection
        self.report["full"]["configs"] = [str(path.relative_to(ROOT)) for path in paths]
        self.save()
        return paths

    def execute_full(self, paths: list[Path]) -> None:
        self.report["full"]["launched"] = True
        self.save()
        for path in paths:
            name = path.stem
            checkpoint = self.train(name, path)
            self.evaluate("full", name, checkpoint)
        self.report["full"]["complete"] = True
        self.save()

    def main(self) -> None:
        paths = self.prepare_screen()
        self.report["prepared_configs"] = [str(path.relative_to(ROOT)) for path in paths.values()]
        self.save()
        if self.args.prepare_only:
            print(json.dumps(self.report, indent=2))
            return

        winner, checkpoint = self.run_screen(paths)
        if winner is None or checkpoint is None:
            self.report["decision"] = "stop_no_global_frequency_promotion"
            self.save()
            print(json.dumps(self.report, indent=2))
            return

        selection = {"attention_dim": 32 if winner.endswith("d32") else 40}
        self.report["phase"] = {
            "deferred": True,
            "reason": "Global-frequency benefit must be established in isolation first.",
        }
        self.report["decision"] = f"prepare_full_{winner}"
        full_paths = self.prepare_full(selection)
        if self.args.execute_full:
            self.execute_full(full_paths)
        print(json.dumps(self.report, indent=2))


def safe_run_id(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "_.-" else "-" for character in value)
    if not safe or safe in {".", ".."}:
        raise ValueError("run-id must contain at least one safe character")
    return safe


if __name__ == "__main__":
    Programme(parse_args()).main()
