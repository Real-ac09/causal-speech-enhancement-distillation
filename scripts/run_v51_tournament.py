#!/usr/bin/env python3
"""Run a resumable, guarded V5.1 architecture/loss tournament."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCKED_400 = Path("data/processed/voicebank_demand/metadata/val_v5_locked_400.csv")
PILOT_100 = Path("data/processed/voicebank_demand/metadata/val_v51_search_100.csv")
CURRENT_V5_EVAL = Path("results/metrics/v5/apples_to_apples_locked400/v5_epoch5")


@dataclass(frozen=True)
class Trial:
    name: str
    architecture: str
    magnitude_mode: str | None = None
    channels: int | None = None


ARCHITECTURES = (
    Trial("v5_control", "causal_aux_vq_mamba_v5"),
    Trial("v51_mask_192", "causal_aux_vq_mamba_v51", "bounded_mask", 192),
    Trial("v51_log_192", "causal_aux_vq_mamba_v51", "log_ratio", 192),
    Trial("v51_compressed_192", "causal_aux_vq_mamba_v51", "compressed_residual", 192),
    Trial("v51_log_256", "causal_aux_vq_mamba_v51", "log_ratio", 256),
    Trial("v51_compressed_256", "causal_aux_vq_mamba_v51", "compressed_residual", 256),
)


LOSS_PROFILES: dict[str, dict[str, float]] = {
    # The weights are deliberately separated into distinct hypotheses. A
    # preliminary gradient audit found waveform, compressed-complex, and
    # direct-magnitude objectives to be near duplicates when enabled together.
    "direct_magnitude": {
        "waveform_l1_weight": 0.0,
        "si_sdr_weight": 0.002,
        "stft_weight": 0.1,
        "mel_weight": 0.1,
        "complex_stft_weight": 0.0,
        "magnitude_weight": 1.0,
        "phase_weight": 1.0,
        "group_delay_weight": 1.0,
        "instantaneous_frequency_weight": 2.0,
        "phase_confidence_weight": 0.05,
    },
    "compressed_complex": {
        "waveform_l1_weight": 0.0,
        "si_sdr_weight": 0.002,
        "stft_weight": 0.1,
        "mel_weight": 0.1,
        "complex_stft_weight": 0.5,
        "magnitude_weight": 0.0,
        "phase_weight": 1.0,
        "group_delay_weight": 1.0,
        "instantaneous_frequency_weight": 2.0,
        "phase_confidence_weight": 0.05,
    },
    "magnitude_heavy": {
        "waveform_l1_weight": 0.0,
        "si_sdr_weight": 0.002,
        "stft_weight": 0.05,
        "mel_weight": 0.15,
        "complex_stft_weight": 0.0,
        "magnitude_weight": 1.5,
        "phase_weight": 0.5,
        "group_delay_weight": 0.5,
        "instantaneous_frequency_weight": 1.0,
        "phase_confidence_weight": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("v51_%Y%m%d_%H%M%S"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-final", action="store_true")
    parser.add_argument("--final-seeds", default="515,516,517")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def ensure_pilot_subset() -> None:
    source = ROOT / LOCKED_400
    target = ROOT / PILOT_100
    rows = list(csv.DictReader(source.open()))
    # The locked set is already stratified. Taking every fourth item preserves
    # coverage better than selecting a contiguous, speaker-clustered prefix.
    selected = rows[::4][:100]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)


def model_config(trial: Trial) -> dict[str, Any]:
    model: dict[str, Any] = {
        "architecture": trial.architecture,
        "variant": "teacher",
        "sample_rate": 16000,
        "n_fft": 512,
        "win_length": 320,
        "hop_length": 160,
        "magnitude_power": 0.3,
        "codebook_size": 32,
        "vq_mode": "train_only",
        "use_mamba": True,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
    }
    if trial.magnitude_mode:
        model["magnitude_mode"] = trial.magnitude_mode
    if trial.channels:
        model["channels"] = trial.channels
        model["noise_dim"] = 96
        model["refinement_passes"] = 2
    return model


def loss_config(profile: str, phase_confidence_available: bool = True) -> dict[str, Any]:
    values = dict(LOSS_PROFILES[profile])
    if not phase_confidence_available:
        values["phase_confidence_weight"] = 0.0
    return {
        **values,
        "waveform_charbonnier_eps": 0.001,
        "stft_fft_sizes": [256, 512, 1024],
        "stft_hop_sizes": [64, 128, 256],
        "stft_win_lengths": [256, 512, 1024],
        "stft_spectral_convergence_weight": 1.0,
        "stft_log_mag_weight": 1.0,
        "stft_mag_l1_weight": 0.0,
        "vq_weight": 1.0,
        "noise_prediction_weight": 0.05,
        "noise_prediction_n_fft": 512,
        "noise_prediction_hop_length": 160,
        "noise_prediction_win_length": 320,
        "complex_stft_fft_sizes": [512],
        "complex_stft_hop_sizes": [160],
        "complex_stft_win_lengths": [320],
        "complex_stft_complex_weight": 1.0,
        "complex_stft_log_mag_weight": 0.0,
        "complex_stft_compression_power": 0.3,
        "magnitude_equal_loudness": True,
        "compute_weight": 0.0,
        "tf_detail_n_fft": 512,
        "tf_detail_hop_length": 160,
        "tf_detail_win_length": 320,
        "tf_detail_center": False,
        "tf_detail_magnitude_power": 0.3,
    }


def make_config(
    trial: Trial,
    profile: str,
    experiment: str,
    seed: int,
    checkpoint_root: Path,
    final: bool = False,
) -> dict[str, Any]:
    return {
        "project": {"name": "cn-vqg-speech-enhancement", "seed": seed},
        "data": {
            "train_metadata": "data/processed/voicebank_demand/metadata/train.csv",
            "val_metadata": str(LOCKED_400 if final else PILOT_100),
            **({"full_val_metadata": "data/processed/voicebank_demand/metadata/val.csv"} if final else {}),
            "sample_rate": 16000,
            "chunk_seconds": 4.0,
            "batch_size": 2,
            "num_workers": 4,
        },
        "model": model_config(trial),
        "loss": loss_config(profile, trial.architecture.endswith("v51")),
        "training": {
            "epochs": 40 if final else 5,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "gradient_accumulation_steps": 4,
            "grad_clip_norm": 5.0,
            "precision": "bf16",
            "log_every": 25,
            "val_every": 1,
            "checkpoint_metric": "perceptual_enhanced_pesq",
            "checkpoint_mode": "max",
            "optimizer": {
                "name": "muon",
                "muon_learning_rate": 0.001,
                "adamw_learning_rate": 0.0001,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "muon_weight_decay": 0.00001,
            },
            "lr_scheduler": {
                "name": "reduce_on_plateau",
                "mode": "max",
                "metric": "perceptual_enhanced_pesq",
                "factor": 0.5,
                "patience": 3,
                "min_lr": 0.000001,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 8 if final else 6,
                "min_delta": 0.003,
            },
            "perceptual_validation": {"enabled": True, "max_items": 400 if final else 100},
            "ema": {"enabled": False},
            "metricgan_lite": {"enabled": False},
        },
        "paths": {"checkpoint_dir": str(checkpoint_root), "experiment_name": experiment},
    }


class Tournament:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = ROOT / "results" / "v51_tournament" / args.run_id
        self.config_dir = ROOT / "configs" / "v5" / "generated" / args.run_id
        self.checkpoint_root = Path("checkpoints") / "v51_tournament" / args.run_id
        self.log_dir = self.run_dir / "logs"
        for directory in (self.run_dir, self.config_dir, ROOT / self.checkpoint_root, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.report: dict[str, Any] = {"run_id": args.run_id, "stages": {}, "final_launched": False}

    def save_report(self) -> None:
        (self.run_dir / "decision_report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def write_config(self, name: str, config: dict[str, Any]) -> Path:
        path = self.config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def command(self, name: str, arguments: list[str], allow_failure: bool = False) -> bool:
        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {name}", flush=True)
        log_path = self.log_dir / f"{name}.log"
        with log_path.open("a") as log:
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
        print(f"[{datetime.now().isoformat(timespec='seconds')}] END {name}: {status}", flush=True)
        if status and not allow_failure:
            raise RuntimeError(f"{name} failed with exit code {status}; see {log_path}")
        return status == 0

    def checkpoint_dir(self, experiment: str) -> Path:
        return ROOT / self.checkpoint_root / experiment

    @staticmethod
    def metric(checkpoint_dir: Path) -> dict[str, float]:
        rows = list(csv.DictReader((checkpoint_dir / "metrics.csv").open()))
        # Promotion decisions must use the best observed checkpoint, not the
        # last row. A longer budget can regress even while best.pt remains
        # healthy, and selecting latest previously promoted collapsed models.
        row = max(rows, key=lambda item: float(item["val_perceptual_enhanced_pesq"]))
        names = (
            "val_perceptual_enhanced_pesq",
            "val_perceptual_enhanced_si_sdr",
            "val_perceptual_enhanced_stoi",
            "val_perceptual_enhanced_estoi",
            "val_perceptual_noisy_pesq",
            "val_perceptual_noisy_stoi",
        )
        return {name: float(row[name]) for name in names if row.get(name)}

    def structural_gate(self, trial: Trial, config_path: Path, checkpoint: Path | None, tag: str) -> bool:
        output = self.run_dir / "gates" / f"{tag}.json"
        command = [
            sys.executable,
            "scripts/check_v51_candidate.py",
            "--config", str(config_path),
            "--output", str(output),
            "--device", self.args.device,
        ]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        return self.command(f"gate_{tag}", command, allow_failure=True)

    def train(
        self,
        tag: str,
        config_path: Path,
        experiment: str,
        epochs: int,
        max_train_batches: int | None,
        max_val_batches: int | None,
        resume: bool,
        prefer_best: bool = False,
    ) -> bool:
        checkpoint_directory = self.checkpoint_dir(experiment)
        metrics_path = checkpoint_directory / "metrics.csv"
        latest = checkpoint_directory / "latest.pt"
        if metrics_path.exists():
            completed = [int(row["epoch"]) for row in csv.DictReader(metrics_path.open())]
            if completed and max(completed) >= epochs:
                print(f"SKIP {tag}: epoch {epochs} is already complete", flush=True)
                return True
        if latest.exists():
            resume = True
        command = [sys.executable, "scripts/train.py", "--config", str(config_path), "--device", self.args.device, "--epochs", str(epochs)]
        if max_train_batches:
            command += ["--max-train-batches", str(max_train_batches)]
        if max_val_batches:
            command += ["--max-val-batches", str(max_val_batches)]
        if resume:
            best = checkpoint_directory / "best.pt"
            resume_checkpoint = best if prefer_best and best.exists() else latest
            if not resume_checkpoint.exists():
                print(f"Cannot resume {tag}: missing {resume_checkpoint}", flush=True)
                return False
            command += ["--resume", str(resume_checkpoint)]
        return self.command(tag, command, allow_failure=True)

    def prepare(self) -> dict[str, tuple[Trial, Path]]:
        prepared = {}
        for index, trial in enumerate(ARCHITECTURES):
            experiment = f"arch_{trial.name}"
            config = make_config(trial, "direct_magnitude", experiment, 700 + index, self.checkpoint_root)
            prepared[trial.name] = (trial, self.write_config(experiment, config))
        self.report["prepared_architectures"] = list(prepared)
        self.save_report()
        return prepared

    def run(self) -> None:
        prepared = self.prepare()
        if self.args.prepare_only:
            print(f"Prepared tournament at {self.run_dir}")
            return

        passing: list[str] = []
        structural = {}
        for name, (trial, path) in prepared.items():
            passed = self.structural_gate(trial, path, None, f"{name}_untrained")
            structural[name] = passed
            if passed:
                passing.append(name)
        self.report["stages"]["structural"] = structural
        self.save_report()
        if len(passing) < 2:
            raise RuntimeError("Fewer than two architectures passed structural gates")

        integrated = []
        for name in passing:
            trial, _ = prepared[name]
            experiment = f"integration_{name}"
            config = make_config(trial, "direct_magnitude", experiment, 800 + len(integrated), self.checkpoint_root)
            path = self.write_config(experiment, config)
            if self.train(f"integration_{name}", path, experiment, 1, 20, 5, False):
                checkpoint = self.checkpoint_dir(experiment) / "latest.pt"
                if self.structural_gate(trial, path, checkpoint, f"{name}_integration"):
                    integrated.append(name)
        self.report["stages"]["integration"] = integrated
        self.save_report()

        screened: list[tuple[float, str]] = []
        for name in integrated:
            trial, path = prepared[name]
            experiment = f"arch_{name}"
            if self.train(f"screen_{name}", path, experiment, 1, 250, 50, False):
                metric = self.metric(self.checkpoint_dir(experiment))
                # One-epoch PESQ improvements can carry tiny transient STOI
                # changes. Use the programme's explicit no-harm allowance in
                # this early screen; the final paired-bootstrap gate remains
                # stricter and also checks ESTOI and SI-SDR.
                if (
                    metric["val_perceptual_enhanced_stoi"]
                    >= metric["val_perceptual_noisy_stoi"] - 0.002
                ):
                    screened.append((metric["val_perceptual_enhanced_pesq"], name))
                self.report.setdefault("metrics", {})[f"screen_{name}"] = metric
        screened.sort(reverse=True)
        top_three = [name for _, name in screened[:3]]
        self.report["stages"]["screen_top_three"] = top_three
        self.save_report()
        if not top_three:
            raise RuntimeError("No architecture passed the bounded one-epoch quality screen")

        promoted: list[tuple[float, str]] = []
        for name in top_three:
            trial, path = prepared[name]
            experiment = f"arch_{name}"
            if self.train(f"promote_{name}", path, experiment, 2, 1000, 50, True, True):
                checkpoint_dir = self.checkpoint_dir(experiment)
                audit = self.run_dir / "gradient_audits" / f"architecture_{name}.json"
                self.command(
                    f"audit_{name}",
                    [sys.executable, "scripts/audit_v5_loss_gradients.py", "--checkpoint", str(checkpoint_dir / "latest.pt"), "--config", str(path), "--output", str(audit), "--num-items", "1"],
                    allow_failure=True,
                )
                metric = self.metric(checkpoint_dir)
                promoted.append((metric["val_perceptual_enhanced_pesq"], name))
                self.report.setdefault("metrics", {})[f"promote_{name}"] = metric
        promoted.sort(reverse=True)
        top_two = [name for _, name in promoted[:2]]
        self.report["stages"]["architecture_top_two"] = top_two
        self.save_report()

        semifinal: list[tuple[float, str]] = []
        for name in top_two:
            _, path = prepared[name]
            experiment = f"arch_{name}"
            if self.train(f"semifinal_{name}", path, experiment, 4, 2000, 50, True, True):
                metric = self.metric(self.checkpoint_dir(experiment))
                semifinal.append((metric["val_perceptual_enhanced_pesq"], name))
                self.report.setdefault("metrics", {})[f"semifinal_{name}"] = metric
        semifinal.sort(reverse=True)
        if not semifinal:
            raise RuntimeError("No architecture survived the semifinal")
        architecture_winner = semifinal[0][1]
        winning_trial = prepared[architecture_winner][0]
        self.report["architecture_winner"] = architecture_winner
        self.save_report()

        loss_screen: list[tuple[float, str, Path]] = []
        loss_configs: dict[str, Path] = {}
        for index, profile in enumerate(LOSS_PROFILES):
            experiment = f"loss_{profile}"
            config = make_config(winning_trial, profile, experiment, 900 + index, self.checkpoint_root)
            path = self.write_config(experiment, config)
            loss_configs[profile] = path
            if self.train(f"loss_screen_{profile}", path, experiment, 1, 250, 50, False):
                metric = self.metric(self.checkpoint_dir(experiment))
                loss_screen.append((metric["val_perceptual_enhanced_pesq"], profile, path))
                self.report.setdefault("metrics", {})[f"loss_screen_{profile}"] = metric
        loss_screen.sort(reverse=True)
        loss_top_two = [profile for _, profile, _ in loss_screen[:2]]
        self.report["stages"]["loss_top_two"] = loss_top_two
        self.save_report()

        loss_finalists: list[tuple[float, str]] = []
        for profile in loss_top_two:
            path = loss_configs[profile]
            experiment = f"loss_{profile}"
            if self.train(f"loss_promote_{profile}", path, experiment, 3, 2000, 50, True, True):
                audit_path = self.run_dir / "gradient_audits" / f"loss_{profile}.json"
                audit_ok = self.command(
                    f"loss_audit_{profile}",
                    [sys.executable, "scripts/audit_v5_loss_gradients.py", "--checkpoint", str(self.checkpoint_dir(experiment) / "latest.pt"), "--config", str(path), "--output", str(audit_path), "--num-items", "4", "--threshold", "0.95"],
                    allow_failure=True,
                )
                audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
                if audit_ok and not audit.get("near_duplicate_objectives"):
                    metric = self.metric(self.checkpoint_dir(experiment))
                    loss_finalists.append((metric["val_perceptual_enhanced_pesq"], profile))
                self.report.setdefault("metrics", {})[f"loss_promote_{profile}"] = self.metric(self.checkpoint_dir(experiment))
        loss_finalists.sort(reverse=True)
        if not loss_finalists:
            raise RuntimeError("No loss profile passed the gradient redundancy gate")
        loss_winner = loss_finalists[0][1]
        self.report["loss_winner"] = loss_winner
        self.save_report()

        winner_experiment = f"loss_{loss_winner}"
        winner_config = loss_configs[loss_winner]
        self.train("winner_epoch5", winner_config, winner_experiment, 5, None, 50, True, True)
        winner_checkpoint = self.checkpoint_dir(winner_experiment) / "best.pt"
        self.structural_gate(winning_trial, winner_config, winner_checkpoint, "winner_trained")

        evaluation = self.run_dir / "winner_evaluation"
        self.command(
            "evaluate_winner",
            [sys.executable, "scripts/evaluate.py", "--checkpoint", str(winner_checkpoint), "--metadata", str(PILOT_100), "--output-dir", str(evaluation), "--device", self.args.device],
        )
        comparison = self.run_dir / "bootstrap_vs_current_v5"
        self.command(
            "bootstrap_winner",
            [sys.executable, "scripts/compare_evaluations.py", "--reference", f"current_v5={CURRENT_V5_EVAL}", "--candidate", f"v51_winner={evaluation}", "--output-dir", str(comparison), "--bootstrap-samples", str(self.args.bootstrap_samples)],
        )
        winner_summary = json.loads((evaluation / "summary.json").read_text())["metrics"]
        paired = json.loads((comparison / "comparison.json").read_text())["paired_bootstrap"]["v51_winner"]
        pesq_delta = paired["enhanced_pesq"]
        final_gates = {
            "pesq_at_least_2_40": winner_summary["enhanced_pesq"] >= 2.40,
            "pesq_delta_at_least_0_10": pesq_delta["mean"] >= 0.10,
            "pesq_ci_excludes_zero": pesq_delta["ci95"][0] > 0.0,
            "si_sdr_no_harm": paired["enhanced_si_sdr"]["mean"] >= -0.15,
            "stoi_no_harm": paired["enhanced_stoi"]["mean"] >= -0.002,
            "estoi_no_harm": paired["enhanced_estoi"]["mean"] >= -0.003,
        }
        self.report["winner_evaluation"] = winner_summary
        self.report["paired_bootstrap"] = paired
        self.report["final_training_gates"] = final_gates
        self.save_report()

        if not all(final_gates.values()):
            print("Final training was not launched because one or more evidence gates failed.")
            return
        final_configs = []
        for seed_text in self.args.final_seeds.split(","):
            seed = int(seed_text.strip())
            experiment = f"final_seed_{seed}"
            config = make_config(winning_trial, loss_winner, experiment, seed, self.checkpoint_root, final=True)
            final_configs.append((seed, experiment, self.write_config(experiment, config)))
        self.report["final_configs"] = [str(path) for _, _, path in final_configs]
        self.save_report()
        if not self.args.execute_final:
            print("Winner passed all gates; final configs prepared. Re-run with --execute-final to train.")
            return
        for seed, experiment, path in final_configs:
            if not self.train(f"final_seed_{seed}", path, experiment, 40, None, None, False):
                raise RuntimeError(f"Final seed {seed} failed")
        self.report["final_launched"] = True
        self.report["final_complete"] = True
        self.save_report()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    ensure_pilot_subset()
    Tournament(args).run()


if __name__ == "__main__":
    main()
