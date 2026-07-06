from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class PairedSpeechDataset(Dataset):
    """
    PyTorch dataset for paired noisy/clean speech enhancement data.

    Expected metadata CSV columns:
    - file_id
    - noisy_path
    - clean_path
    - sample_rate
    - num_samples
    - duration_seconds

    Returns:
        {
            "noisy": Tensor [1, T],
            "clean": Tensor [1, T],
            "file_id": str,
            "speaker_id": str,
            "sample_rate": int,
        }
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        project_root: str | Path = ".",
        sample_rate: int = 16000,
        chunk_seconds: Optional[float] = 4.0,
        random_crop: bool = True,
        peak_normalize: bool = False,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.project_root = Path(project_root)
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.random_crop = random_crop
        self.peak_normalize = peak_normalize

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        self.metadata = pd.read_csv(self.metadata_csv)

        required_columns = {
            "file_id",
            "noisy_path",
            "clean_path",
            "sample_rate",
            "num_samples",
            "duration_seconds",
        }

        missing_columns = required_columns - set(self.metadata.columns)
        if missing_columns:
            raise ValueError(
                f"Metadata CSV is missing required columns: {sorted(missing_columns)}"
            )

        if len(self.metadata) == 0:
            raise ValueError(f"Metadata CSV is empty: {self.metadata_csv}")

        if self.chunk_seconds is not None:
            self.chunk_samples = int(round(self.chunk_seconds * self.sample_rate))
        else:
            self.chunk_samples = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        row = self.metadata.iloc[index]

        noisy_path = self._resolve_path(row["noisy_path"])
        clean_path = self._resolve_path(row["clean_path"])

        noisy = self._load_audio(noisy_path)
        clean = self._load_audio(clean_path)

        noisy, clean = self._align_lengths(noisy, clean)

        if self.peak_normalize:
            noisy, clean = self._peak_normalize_pair(noisy, clean)

        if self.chunk_samples is not None:
            noisy, clean = self._crop_or_pad_pair(noisy, clean, self.chunk_samples)

        return {
            "noisy": noisy,
            "clean": clean,
            "file_id": str(row["file_id"]),
            "speaker_id": str(row["speaker_id"]) if "speaker_id" in row else "unknown",
            "sample_rate": self.sample_rate,
        }

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)

        if path.is_absolute():
            return path

        return self.project_root / path

    def _load_audio(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sr = torchaudio.load(path)

        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [channels, samples], got {waveform.shape}")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform.float()

    @staticmethod
    def _align_lengths(
        noisy: torch.Tensor,
        clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        min_len = min(noisy.shape[-1], clean.shape[-1])
        return noisy[..., :min_len], clean[..., :min_len]

    @staticmethod
    def _peak_normalize_pair(
        noisy: torch.Tensor,
        clean: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak = torch.maximum(noisy.abs().max(), clean.abs().max())
        peak = torch.clamp(peak, min=eps)

        return noisy / peak, clean / peak

    def _crop_or_pad_pair(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        chunk_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_samples = noisy.shape[-1]

        if current_samples == chunk_samples:
            return noisy, clean

        if current_samples > chunk_samples:
            if self.random_crop:
                start = random.randint(0, current_samples - chunk_samples)
            else:
                start = (current_samples - chunk_samples) // 2

            end = start + chunk_samples
            return noisy[..., start:end], clean[..., start:end]

        pad_amount = chunk_samples - current_samples

        noisy = F.pad(noisy, (0, pad_amount))
        clean = F.pad(clean, (0, pad_amount))

        return noisy, clean


def speech_enhancement_collate_fn(batch):
    """
    Collate function for batches where all items have already been cropped/padded
    to the same length.
    """

    noisy = torch.stack([item["noisy"] for item in batch], dim=0)
    clean = torch.stack([item["clean"] for item in batch], dim=0)

    file_ids = [item["file_id"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    sample_rates = [item["sample_rate"] for item in batch]

    return {
        "noisy": noisy,
        "clean": clean,
        "file_id": file_ids,
        "speaker_id": speaker_ids,
        "sample_rate": sample_rates,
    }

