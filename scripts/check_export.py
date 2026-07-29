#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
from torch import nn

from cnvqg.models.factory import build_model


class EnhancedOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform).enhanced


def attempt(name: str, operation, report: dict) -> None:
    try:
        operation()
        report[name] = {"success": True}
    except Exception as error:
        report[name] = {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose deployment export support.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16000)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapper = EnhancedOnly(model.eval())
    example = torch.randn(1, 1, args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(args.checkpoint), "torch_version": torch.__version__}

    attempt(
        "torchscript_trace",
        lambda: torch.jit.trace(wrapper, example, check_trace=False).save(
            str(args.output_dir / "model_traced.pt")
        ),
        report,
    )

    def export_program() -> None:
        program = torch.export.export(
            wrapper,
            (example,),
            dynamic_shapes={"waveform": {2: torch.export.Dim("samples", min=512)}},
        )
        torch.export.save(program, args.output_dir / "model_exported.pt2")

    attempt("torch_export", export_program, report)
    (args.output_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
