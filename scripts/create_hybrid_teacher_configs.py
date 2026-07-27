from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


BASE_CONFIG = Path(
    "configs/optimised/train_hybrid_tf_phase4c_magphase.yaml"
)

OUTPUT_DIR = Path("configs/teachers")


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def size_matches(path: Path, size: str) -> bool:
    text = str(path).lower()
    return re.search(
        rf"(^|[/_-]){re.escape(size)}([/_-]|$)",
        text,
    ) is not None


def candidate_score(path: Path) -> int:
    name = str(path).lower()
    score = 0

    if "mamba_true" in name:
        score += 100

    if "opt_sched" in name:
        score += 50

    if "adjusted" in name:
        score += 10

    if "fallback" in name or "old" in name:
        score -= 100

    return score


def inspect_candidate(
    path: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        checkpoint = load_checkpoint(path)
    except Exception as exc:
        return False, f"could not load: {exc}", None

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    if not model_config:
        return False, "checkpoint has no saved model config", None

    if not bool(model_config.get("use_mamba", False)):
        return False, "saved config does not use Mamba", model_config

    if "model_state_dict" not in checkpoint:
        return False, "checkpoint has no model_state_dict", model_config

    return True, "valid real-Mamba checkpoint", model_config


def find_checkpoint(size: str) -> Path:
    env_name = f"CNVQG_{size.upper()}_CHECKPOINT"
    explicit = os.environ.get(env_name)

    if explicit:
        path = Path(explicit)

        if not path.is_file():
            raise FileNotFoundError(
                f"{env_name} points to a missing file: {path}"
            )

        valid, reason, _ = inspect_candidate(path)

        if not valid:
            raise RuntimeError(
                f"{path} is not suitable: {reason}"
            )

        print(f"{size}: using explicit checkpoint {path}")
        return path

    candidates = [
        path
        for path in Path("checkpoints").glob("*/best.pt")
        if size_matches(path, size)
    ]

    valid_candidates: list[Path] = []

    print(f"\n{size} checkpoint candidates:")

    for path in sorted(candidates):
        valid, reason, _ = inspect_candidate(path)
        status = "VALID" if valid else "REJECTED"
        print(f"  [{status}] {path} — {reason}")

        if valid:
            valid_candidates.append(path)

    if not valid_candidates:
        raise FileNotFoundError(
            f"No valid real-Mamba {size} checkpoint was found.\n"
            f"Set {env_name}=path/to/best.pt after training the "
            f"{size} waveform model."
        )

    valid_candidates.sort(
        key=candidate_score,
        reverse=True,
    )

    selected = valid_candidates[0]
    print(f"{size}: automatically selected {selected}")
    return selected


def approximate_state_size(checkpoint_path: Path) -> int:
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint["model_state_dict"]

    return sum(
        tensor.numel()
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )


def create_config(
    base: dict[str, Any],
    size: str,
    checkpoint: Path,
) -> Path:
    config = deepcopy(base)

    experiment = f"cnvqg_hybrid_tf_{size}_teacher_stage1"

    config["paths"]["experiment_name"] = experiment

    # The backbone configuration is read directly from its checkpoint.
    config["model"] = {
        "architecture": "hybrid_tf_refiner",
        "waveform_checkpoint": str(checkpoint),
        "waveform_model_config": None,
        "freeze_waveform": True,
        "sample_rate": 16000,
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,

        # Keep this identical for 25M and 50M during Stage 1.
        "hidden_dim": 64,
        "temporal_layers": 4,
        "use_mamba": True,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,

        "predict_phase": True,
        "gain_scale": 0.20,
        "phase_scale": 0.08,
        "center": True,
    }

    # The hybrid architecture internally loads its waveform checkpoint.
    config["training"].pop("init_checkpoint", None)

    config["training"]["epochs"] = 10
    config["training"]["learning_rate"] = 0.00005
    config["training"]["mixed_precision"] = False

    config["training"]["lr_scheduler"] = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 2,
        "min_lr": 0.000005,
    }

    config["training"]["early_stopping"] = {
        "enabled": True,
        "patience": 4,
        "min_delta": 0.0001,
    }

    config["training"]["ema"] = {
        "enabled": False,
        "decay": 0.999,
    }

    # Establish the architecture result before adding MetricGAN.
    config["training"]["metricgan_lite"] = {
        "enabled": False,
    }

    config["loss"]["mel_weight"] = 0.20
    config["loss"]["complex_stft_weight"] = 0.0

    output = OUTPUT_DIR / f"train_hybrid_tf_{size}_teacher_stage1.yaml"
    output.write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    return output


def main() -> None:
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(
            f"Missing base hybrid config: {BASE_CONFIG}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(BASE_CONFIG.read_text())

    for size in ("25m", "50m"):
        checkpoint = find_checkpoint(size)
        output = create_config(base, size, checkpoint)
        state_size = approximate_state_size(checkpoint)

        print(f"\nCreated {output}")
        print(f"  waveform checkpoint: {checkpoint}")
        print(f"  checkpoint state elements: {state_size:,}")
        print("  waveform backbone frozen: yes")
        print("  TF refiner: hidden_dim=64, layers=4")


if __name__ == "__main__":
    main()
