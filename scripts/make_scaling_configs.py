from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/ablations/train_residual_mamba_fixed_scale.yaml")
OUT_DIR = Path("configs/scaling")

OUT_DIR.mkdir(parents=True, exist_ok=True)

base = yaml.safe_load(BASE_CONFIG.read_text())

variants = {
    "1m": {
        "encoder_channels": 64,
        "latent_dim": 128,
        "speech_dim": 128,
        "noise_dim": 64,
        "codebook_size": 256,
        "temporal_layers": 4,
    },
    "2m": {
        "encoder_channels": 80,
        "latent_dim": 160,
        "speech_dim": 160,
        "noise_dim": 80,
        "codebook_size": 512,
        "temporal_layers": 4,
    },
    "5m": {
        "encoder_channels": 128,
        "latent_dim": 256,
        "speech_dim": 256,
        "noise_dim": 128,
        "codebook_size": 512,
        "temporal_layers": 6,
    },
    "10m": {
        "encoder_channels": 160,
        "latent_dim": 320,
        "speech_dim": 320,
        "noise_dim": 160,
        "codebook_size": 1024,
        "temporal_layers": 8,
    },
    "25m": {
        "encoder_channels": 256,
        "latent_dim": 512,
        "speech_dim": 512,
        "noise_dim": 256,
        "codebook_size": 1024,
        "temporal_layers": 10,
    },
    "50m": {
        "encoder_channels": 384,
        "latent_dim": 768,
        "speech_dim": 768,
        "noise_dim": 384,
        "codebook_size": 2048,
        "temporal_layers": 12,
    },
}

for name, model_updates in variants.items():
    cfg = deepcopy(base)

    cfg["paths"]["experiment_name"] = f"cnvqg_mamba_{name}"

    cfg["model"].update(model_updates)

    cfg["model"]["use_mamba"] = True
    cfg["model"]["use_residual"] = True
    cfg["model"]["use_vq"] = True
    cfg["model"]["use_noise_conditioning"] = True
    cfg["model"]["use_temporal"] = True
    cfg["model"]["residual_scale_init"] = 0.2
    cfg["model"]["learn_residual_scale"] = False

    cfg["training"]["mixed_precision"] = False
    cfg["training"]["learning_rate"] = 0.0001

    # Larger models may need smaller batches. This only matters if train.py reads batch_size.
    if "batch_size" in cfg["training"]:
        if name in {"25m", "50m"}:
            cfg["training"]["batch_size"] = 1
        elif name == "10m":
            cfg["training"]["batch_size"] = min(int(cfg["training"]["batch_size"]), 2)
        else:
            cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])

    out = OUT_DIR / f"train_mamba_{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("Wrote", out)
