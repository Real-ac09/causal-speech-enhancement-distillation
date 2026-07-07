from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from cnvqg.models import CNVQGModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    model = CNVQGModel(**cfg["model"])

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Config: {args.config}")
    print(f"Experiment: {cfg['paths']['experiment_name']}")
    print(f"Uses Mamba: {model.temporal.uses_mamba}")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = torch.randn(1, 1, 64000, device=device)

    with torch.no_grad():
        y = model(x)

    print(f"Input: {x.shape}")
    print(f"Enhanced: {y.enhanced.shape}")
    print("Forward pass OK")


if __name__ == "__main__":
    main()
