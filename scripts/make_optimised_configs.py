from __future__ import annotations

from pathlib import Path
import yaml

SOURCE_CONFIGS = {
    "1m": Path("configs/scaling/train_mamba_1m.yaml"),
    "5m": Path("configs/scaling/train_mamba_5m.yaml"),
    "10m": Path("configs/scaling/train_mamba_10m.yaml"),
}

OUT_DIR = Path("configs/optimised")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_epochs(training: dict, epochs: int) -> None:
    for key in ["epochs", "num_epochs", "max_epochs"]:
        if key in training:
            training[key] = epochs
            return

    # If your train.py uses another name, this will be harmless but visible in config.
    training["epochs"] = epochs


for size, src in SOURCE_CONFIGS.items():
    cfg = yaml.safe_load(src.read_text())

    cfg["paths"]["experiment_name"] = f"cnvqg_mamba_{size}_opt_long"

    cfg["model"]["use_mamba"] = True
    cfg["model"]["use_residual"] = True
    cfg["model"]["use_vq"] = True
    cfg["model"]["use_noise_conditioning"] = True
    cfg["model"]["use_temporal"] = True
    cfg["model"]["residual_scale_init"] = 0.2
    cfg["model"]["learn_residual_scale"] = False

    # Longer training.
    set_epochs(cfg["training"], 20)

    # Keep LR stable for now.
    cfg["training"]["learning_rate"] = 0.0001
    cfg["training"]["mixed_precision"] = False

    out = OUT_DIR / f"train_mamba_{size}_opt_long.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Wrote", out)
