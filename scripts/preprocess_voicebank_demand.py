#!/usr/bin/env python3

"""
Preprocess VoiceBank + DEMAND for speech enhancement.

Expected common raw dataset layout:

data/raw/voicebank_demand/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/

The script is intentionally flexible and searches for folders containing:
- clean + train
- noisy + train
- clean + test
- noisy + test

It creates:

data/processed/voicebank_demand/
├── train/
│   ├── clean/
│   └── noisy/
├── val/
│   ├── clean/
│   └── noisy/
├── test/
│   ├── clean/
│   └── noisy/
└── metadata/
    ├── train.csv
    ├── val.csv
    └── test.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm


@dataclass(frozen=True)
class AudioPair:
    file_id: str
    noisy_path: Path
    clean_path: Path
    speaker_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess VoiceBank + DEMAND noisy/clean WAV pairs."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/voicebank_demand"),
        help="Path to extracted raw VoiceBank + DEMAND folder.",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/voicebank_demand"),
        help="Output folder for processed dataset.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of training speakers to reserve for validation.",
    )

    parser.add_argument(
        "--val-speakers",
        type=str,
        default="",
        help="Optional comma-separated speaker IDs for validation, e.g. p226,p287.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for speaker-level validation split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed WAV files.",
    )

    return parser.parse_args()


def find_wav_dirs(root: Path) -> List[Path]:
    wav_dirs: List[Path] = []

    for directory in root.rglob("*"):
        if directory.is_dir():
            has_wavs = any(directory.glob("*.wav"))
            if has_wavs:
                wav_dirs.append(directory)

    return sorted(wav_dirs)


def pick_dataset_dir(wav_dirs: Sequence[Path], required_terms: Sequence[str]) -> Path:
    matches: List[Path] = []

    for directory in wav_dirs:
        name = str(directory).lower()
        if all(term in name for term in required_terms):
            matches.append(directory)

    if not matches:
        searched = "\n".join(str(path) for path in wav_dirs)
        raise FileNotFoundError(
            f"Could not find folder containing terms: {required_terms}\n\n"
            f"WAV folders found:\n{searched}"
        )

    # Prefer the shortest path if multiple match.
    return sorted(matches, key=lambda path: len(str(path)))[0]


def collect_wavs(directory: Path) -> Dict[str, Path]:
    wavs: Dict[str, Path] = {}

    for wav_path in sorted(directory.rglob("*.wav")):
        wavs[wav_path.stem] = wav_path

    return wavs


def speaker_from_file_id(file_id: str) -> str:
    # VoiceBank filenames are usually like: p226_001
    if "_" in file_id:
        return file_id.split("_")[0]

    return "unknown"


def pair_files(noisy_files: Dict[str, Path], clean_files: Dict[str, Path]) -> List[AudioPair]:
    common_ids = sorted(set(noisy_files) & set(clean_files))

    missing_clean = sorted(set(noisy_files) - set(clean_files))
    missing_noisy = sorted(set(clean_files) - set(noisy_files))

    if missing_clean:
        print(f"[WARN] {len(missing_clean)} noisy files have no matching clean file.")

    if missing_noisy:
        print(f"[WARN] {len(missing_noisy)} clean files have no matching noisy file.")

    pairs = [
        AudioPair(
            file_id=file_id,
            noisy_path=noisy_files[file_id],
            clean_path=clean_files[file_id],
            speaker_id=speaker_from_file_id(file_id),
        )
        for file_id in common_ids
    ]

    if not pairs:
        raise RuntimeError("No matching noisy/clean pairs were found.")

    return pairs


def split_train_val_by_speaker(
    pairs: Sequence[AudioPair],
    val_fraction: float,
    seed: int,
    explicit_val_speakers: Optional[Sequence[str]] = None,
) -> Tuple[List[AudioPair], List[AudioPair]]:
    speakers = sorted({pair.speaker_id for pair in pairs})

    if explicit_val_speakers:
        val_speakers = set(explicit_val_speakers)
    else:
        rng = random.Random(seed)
        shuffled_speakers = speakers[:]
        rng.shuffle(shuffled_speakers)

        n_val = max(1, int(round(len(shuffled_speakers) * val_fraction)))
        n_val = min(n_val, len(shuffled_speakers) - 1)

        val_speakers = set(shuffled_speakers[:n_val])

    train_pairs = [pair for pair in pairs if pair.speaker_id not in val_speakers]
    val_pairs = [pair for pair in pairs if pair.speaker_id in val_speakers]

    if not train_pairs:
        raise RuntimeError("Training split is empty. Reduce validation speakers/fraction.")

    if not val_pairs:
        raise RuntimeError("Validation split is empty. Increase validation speakers/fraction.")

    print(f"Training speakers: {sorted({pair.speaker_id for pair in train_pairs})}")
    print(f"Validation speakers: {sorted({pair.speaker_id for pair in val_pairs})}")

    return train_pairs, val_pairs


def load_resample_mono(path: Path, sample_rate: int) -> Tuple[torch.Tensor, int]:
    waveform, original_sr = torchaudio.load(path)

    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {waveform.shape}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if original_sr != sample_rate:
        resampler = Resample(orig_freq=original_sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.clamp(-1.0, 1.0)

    return waveform, sample_rate


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(
        str(path),
        waveform,
        sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def process_split(
    split_name: str,
    pairs: Sequence[AudioPair],
    processed_dir: Path,
    sample_rate: int,
    overwrite: bool,
) -> List[Dict[str, str]]:
    split_dir = processed_dir / split_name
    noisy_out_dir = split_dir / "noisy"
    clean_out_dir = split_dir / "clean"

    metadata_rows: List[Dict[str, str]] = []

    for pair in tqdm(pairs, desc=f"Processing {split_name}"):
        noisy_out = noisy_out_dir / f"{pair.file_id}.wav"
        clean_out = clean_out_dir / f"{pair.file_id}.wav"

        if overwrite or not noisy_out.exists():
            noisy_waveform, sr = load_resample_mono(pair.noisy_path, sample_rate)
            save_wav(noisy_out, noisy_waveform, sr)
        else:
            noisy_waveform, sr = torchaudio.load(noisy_out)

        if overwrite or not clean_out.exists():
            clean_waveform, sr = load_resample_mono(pair.clean_path, sample_rate)
            save_wav(clean_out, clean_waveform, sr)
        else:
            clean_waveform, sr = torchaudio.load(clean_out)

        num_samples = min(noisy_waveform.shape[-1], clean_waveform.shape[-1])
        duration = num_samples / sample_rate

        metadata_rows.append(
            {
                "split": split_name,
                "file_id": pair.file_id,
                "speaker_id": pair.speaker_id,
                "noisy_path": relative_to_project(noisy_out),
                "clean_path": relative_to_project(clean_out),
                "sample_rate": str(sample_rate),
                "num_samples": str(num_samples),
                "duration_seconds": f"{duration:.6f}",
            }
        )

    return metadata_rows


def write_metadata(processed_dir: Path, split_name: str, rows: Sequence[Dict[str, str]]) -> None:
    metadata_dir = processed_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metadata_dir / f"{split_name}.csv"

    fieldnames = [
        "split",
        "file_id",
        "speaker_id",
        "noisy_path",
        "clean_path",
        "sample_rate",
        "num_samples",
        "duration_seconds",
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote metadata: {csv_path} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()

    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset folder does not exist: {raw_dir}\n"
            f"Place the extracted VoiceBank + DEMAND dataset there first."
        )

    wav_dirs = find_wav_dirs(raw_dir)

    if not wav_dirs:
        raise FileNotFoundError(f"No WAV folders found inside: {raw_dir}")

    clean_train_dir = pick_dataset_dir(wav_dirs, ["clean", "train"])
    noisy_train_dir = pick_dataset_dir(wav_dirs, ["noisy", "train"])
    clean_test_dir = pick_dataset_dir(wav_dirs, ["clean", "test"])
    noisy_test_dir = pick_dataset_dir(wav_dirs, ["noisy", "test"])

    print("Detected folders:")
    print(f"  clean train: {clean_train_dir}")
    print(f"  noisy train: {noisy_train_dir}")
    print(f"  clean test:  {clean_test_dir}")
    print(f"  noisy test:  {noisy_test_dir}")

    train_pairs_all = pair_files(
        noisy_files=collect_wavs(noisy_train_dir),
        clean_files=collect_wavs(clean_train_dir),
    )

    test_pairs = pair_files(
        noisy_files=collect_wavs(noisy_test_dir),
        clean_files=collect_wavs(clean_test_dir),
    )

    explicit_val_speakers = None
    if args.val_speakers.strip():
        explicit_val_speakers = [
            speaker.strip()
            for speaker in args.val_speakers.split(",")
            if speaker.strip()
        ]

    train_pairs, val_pairs = split_train_val_by_speaker(
        pairs=train_pairs_all,
        val_fraction=args.val_fraction,
        seed=args.seed,
        explicit_val_speakers=explicit_val_speakers,
    )

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    for split_name, split_pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        rows = process_split(
            split_name=split_name,
            pairs=split_pairs,
            processed_dir=processed_dir,
            sample_rate=args.sample_rate,
            overwrite=args.overwrite,
        )

        write_metadata(
            processed_dir=processed_dir,
            split_name=split_name,
            rows=rows,
        )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
