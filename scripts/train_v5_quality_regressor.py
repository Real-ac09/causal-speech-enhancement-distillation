#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from cnvqg.data import PairedSpeechDataset
from cnvqg.metrics import safe_pesq
from train import QualityDiscriminator


def pad_pair(candidate: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(candidate.numel(), clean.numel())
    return F.pad(candidate, (0, length - candidate.numel())), F.pad(clean, (0, length - clean.numel()))


def examples(dataset: PairedSpeechDataset, enhanced_dir: Path | None, limit: int):
    for index in range(min(limit, len(dataset))):
        item = dataset[index]
        noisy, clean = item["noisy"], item["clean"]
        yield noisy, clean
        yield clean, clean
        # Interpolated candidates fill the quality range and stabilize the
        # calibration before saved generator outputs become available.
        yield 0.5 * noisy + 0.5 * clean, clean
        if enhanced_dir is not None:
            path = enhanced_dir / f"{item['file_id']}.wav"
            if path.exists():
                audio, rate = sf.read(path, dtype="float32")
                if rate != 16000:
                    raise ValueError(f"Expected 16 kHz enhanced audio: {path}")
                yield torch.from_numpy(audio).view(1, -1), clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and gate the V5 PESQ regressor.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--enhanced-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=1337)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedSpeechDataset(args.metadata, chunk_seconds=4.0, random_crop=False)
    records = []
    cache_path = args.output.with_suffix(".targets.pt")
    if cache_path.exists():
        records = torch.load(cache_path, weights_only=False)
    else:
        for candidate, clean in examples(dataset, args.enhanced_dir, args.max_files):
            candidate, clean = pad_pair(candidate.squeeze(0), clean.squeeze(0))
            score = safe_pesq(clean.numpy(), candidate.numpy(), 16000)
            if score is not None:
                records.append((candidate.unsqueeze(0), clean.unsqueeze(0), float(score)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, cache_path)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(records), generator=generator).tolist()
    split = max(1, int(0.8 * len(order)))
    train_ids, validation_ids = order[:split], order[split:]
    network = QualityDiscriminator().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(args.epochs):
        network.train()
        random.shuffle(train_ids)
        losses = []
        for index in train_ids:
            candidate, clean, score = records[index]
            candidate, clean = candidate.to(device), clean.to(device)
            target = candidate.new_tensor([min(1.0, max(0.0, (score - 1.0) / 3.5))])
            prediction = torch.sigmoid(network(candidate.unsqueeze(0), clean.unsqueeze(0)))
            loss = F.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}")
    network.eval()
    predicted, target = [], []
    with torch.inference_mode():
        for index in validation_ids:
            candidate, clean, score = records[index]
            value = 1.0 + 3.5 * torch.sigmoid(
                network(candidate.to(device).unsqueeze(0), clean.to(device).unsqueeze(0))
            ).item()
            predicted.append(value)
            target.append(score)
    pearson = float(np.corrcoef(predicted, target)[0, 1])
    mae = float(np.mean(np.abs(np.asarray(predicted) - np.asarray(target))))
    calibration = {"pearson": pearson, "mae": mae, "examples": len(validation_ids)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": network.state_dict(), "calibration": calibration}, args.output)
    print("calibration", calibration)
    if pearson < 0.80 or mae > 0.12:
        raise SystemExit("Calibration gate failed; generator gradients remain disabled.")


if __name__ == "__main__":
    main()
