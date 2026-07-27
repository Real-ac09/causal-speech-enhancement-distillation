#!/usr/bin/env python3
"""Select and materialize leakage-safe V16 oracle training mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
POINTER_OID = re.compile(r"oid sha256:([0-9a-f]{64})")
POINTER_SIZE = re.compile(r"size (\d+)")
READER = re.compile(r"_reader_(\d+)_")
SNR_LEVELS = (-5.0, 0.0, 5.0, 10.0, 15.0)
CLEAN_RMS_LEVELS = (-35.0, -30.0, -25.0, -20.0)
SAMPLE_RATE = 16_000
DURATION_SECONDS = 4.0
SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def _pointer(path: Path) -> tuple[str, int] | None:
    if path.stat().st_size > 1024:
        return None
    text = path.read_text(errors="replace")
    oid = POINTER_OID.search(text)
    size = POINTER_SIZE.search(text)
    if oid is None or size is None:
        return None
    return oid.group(1), int(size.group(1))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _split_items(
    items: list[dict[str, Any]],
    *,
    training: int,
    calibration: int,
) -> list[dict[str, Any]]:
    required = training + calibration
    if len(items) < required:
        raise RuntimeError(
            f"Selected {len(items)} items but {required} are required"
        )
    selected = items[:required]
    for index, item in enumerate(selected):
        item["oracle_role"] = (
            "training" if index < training else "training_calibration"
        )
    return selected


def _select_voice(
    metadata_path: Path,
    *,
    training: int,
    calibration: int,
    seed: int,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(metadata_path)
    if frame["file_id"].duplicated().any():
        raise ValueError("VoiceBank training file IDs must be unique")
    if "speaker_id" not in frame.columns:
        raise ValueError("VoiceBank metadata requires speaker_id")
    frame["_score"] = frame["file_id"].map(lambda value: _score(str(value), seed))
    speaker_order = sorted(
        frame["speaker_id"].astype(str).unique(),
        key=lambda speaker: _score(speaker, seed + 1),
    )
    calibration_speakers: set[str] = set()
    calibration_capacity = 0
    speaker_counts = frame["speaker_id"].astype(str).value_counts()
    for speaker in speaker_order:
        calibration_speakers.add(speaker)
        calibration_capacity += int(speaker_counts[speaker])
        if len(calibration_speakers) >= 2 and calibration_capacity >= calibration:
            break
    speaker_values = frame["speaker_id"].astype(str)
    calibration_pool = frame[
        speaker_values.isin(calibration_speakers)
    ].sort_values("_score")
    training_pool = frame[
        ~speaker_values.isin(calibration_speakers)
    ].sort_values("_score")
    if len(training_pool) < training or len(calibration_pool) < calibration:
        raise RuntimeError(
            "VoiceBank speaker-disjoint split does not have enough items"
        )
    training_items = (
        training_pool.iloc[:training]
        .drop(columns=["_score"])
        .to_dict(orient="records")
    )
    calibration_items = (
        calibration_pool.iloc[:calibration]
        .drop(columns=["_score"])
        .to_dict(orient="records")
    )
    for item in training_items:
        item["oracle_role"] = "training"
    for item in calibration_items:
        item["oracle_role"] = "training_calibration"
    return [*training_items, *calibration_items]


def _select_dns(
    dataset_root: Path,
    exclusion: dict[str, Any],
    *,
    training: int,
    calibration: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_clean = {
        str(item["lfs_oid_sha256"])
        for item in exclusion["clean_assets"]
    }
    excluded_noise = {
        str(item["lfs_oid_sha256"])
        for item in exclusion["noise_assets"]
    }
    clean_candidates: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "datasets/clean").glob("*.wav")):
        pointer = _pointer(path)
        reader = READER.search(path.name)
        if pointer is None or reader is None:
            continue
        oid, size = pointer
        if oid in excluded_clean or size < SAMPLES * 2 or size > 8_000_000:
            continue
        relative = str(path.relative_to(dataset_root))
        clean_candidates.append(
            {
                "path": relative,
                "lfs_oid_sha256": oid,
                "lfs_size": size,
                "reader_id": reader.group(1),
                "score": _score(relative, seed),
            }
        )
    clean_candidates.sort(key=lambda item: str(item["score"]))
    unique_clean: list[dict[str, Any]] = []
    readers: set[str] = set()
    for item in clean_candidates:
        reader = str(item["reader_id"])
        if reader in readers:
            continue
        readers.add(reader)
        unique_clean.append(item)

    noise_candidates: list[dict[str, Any]] = []
    for path in sorted((dataset_root / "datasets/noise").glob("*.wav")):
        pointer = _pointer(path)
        if pointer is None:
            continue
        oid, size = pointer
        if oid in excluded_noise or size < SAMPLES * 2 or size > 16_000_000:
            continue
        relative = str(path.relative_to(dataset_root))
        noise_candidates.append(
            {
                "path": relative,
                "lfs_oid_sha256": oid,
                "lfs_size": size,
                "score": _score(relative, seed + 1),
            }
        )
    noise_candidates.sort(key=lambda item: str(item["score"]))
    clean = _split_items(
        unique_clean,
        training=training,
        calibration=calibration,
    )
    noise = _split_items(
        noise_candidates,
        training=training,
        calibration=calibration,
    )
    return clean, noise


def _load_mono(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if rate != SAMPLE_RATE:
        values = librosa.resample(
            values,
            orig_sr=rate,
            target_sr=SAMPLE_RATE,
        ).astype(np.float32)
    return values


def _fixed_segment(
    values: np.ndarray,
    *,
    selector: int,
) -> np.ndarray:
    if len(values) < SAMPLES:
        return np.pad(values, (0, SAMPLES - len(values)))
    maximum = len(values) - SAMPLES
    start = selector % (maximum + 1) if maximum else 0
    return values[start : start + SAMPLES].copy()


def _rms(values: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.square(values.astype(np.float64))) + 1e-12)
    )


def _write_pair(
    *,
    output_root: Path,
    role: str,
    file_id: str,
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[Path, Path]:
    role_name = "train" if role == "training" else "calibration"
    clean_path = output_root / "audio" / role_name / "clean" / f"{file_id}.wav"
    noisy_path = output_root / "audio" / role_name / "noisy" / f"{file_id}.wav"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(clean_path, clean, SAMPLE_RATE, subtype="PCM_16")
    sf.write(noisy_path, noisy, SAMPLE_RATE, subtype="PCM_16")
    return clean_path, noisy_path


def _materialize_voice(
    selected: list[dict[str, Any]],
    *,
    output_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, source in enumerate(selected):
        noisy = _load_mono(_resolve(Path(str(source["noisy_path"]))))
        clean = _load_mono(_resolve(Path(str(source["clean_path"]))))
        length = min(len(noisy), len(clean))
        noisy = noisy[:length]
        clean = clean[:length]
        selector = int(_score(str(source["file_id"]), seed)[:12], 16)
        if length >= SAMPLES:
            maximum = length - SAMPLES
            start = selector % (maximum + 1) if maximum else 0
            noisy = noisy[start : start + SAMPLES].copy()
            clean = clean[start : start + SAMPLES].copy()
        else:
            noisy = np.pad(noisy, (0, SAMPLES - length))
            clean = np.pad(clean, (0, SAMPLES - length))
        file_id = f"v16_vb_{index:04d}_{source['file_id']}"
        clean_path, noisy_path = _write_pair(
            output_root=output_root,
            role=str(source["oracle_role"]),
            file_id=file_id,
            clean=clean,
            noisy=noisy,
        )
        rows.append(
            {
                "oracle_role": source["oracle_role"],
                "domain": "voicebank_demand",
                "file_id": file_id,
                "speaker_id": source.get("speaker_id", "unknown"),
                "noisy_path": _relative(noisy_path),
                "clean_path": _relative(clean_path),
                "sample_rate": SAMPLE_RATE,
                "num_samples": SAMPLES,
                "duration_seconds": DURATION_SECONDS,
                "source_file_id": source["file_id"],
            }
        )
    return rows


def _materialize_dns(
    clean_assets: list[dict[str, Any]],
    noise_assets: list[dict[str, Any]],
    *,
    dataset_root: Path,
    output_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, (clean_asset, noise_asset) in enumerate(
        zip(clean_assets, noise_assets)
    ):
        clean_source = dataset_root / str(clean_asset["path"])
        noise_source = dataset_root / str(noise_asset["path"])
        if _sha256(clean_source) != clean_asset["lfs_oid_sha256"]:
            raise ValueError(f"Clean LFS object mismatch: {clean_source}")
        if _sha256(noise_source) != noise_asset["lfs_oid_sha256"]:
            raise ValueError(f"Noise LFS object mismatch: {noise_source}")
        selector = int(_score(str(index), seed)[:12], 16)
        clean = _fixed_segment(
            _load_mono(clean_source),
            selector=selector,
        )
        noise = _fixed_segment(
            _load_mono(noise_source),
            selector=selector // 7,
        )
        target_clean_dbfs = CLEAN_RMS_LEVELS[
            (index // len(SNR_LEVELS)) % len(CLEAN_RMS_LEVELS)
        ]
        snr_db = SNR_LEVELS[index % len(SNR_LEVELS)]
        clean *= 10.0 ** (target_clean_dbfs / 20.0) / _rms(clean)
        noise *= (
            _rms(clean) / (10.0 ** (snr_db / 20.0)) / _rms(noise)
        )
        noisy = clean + noise
        peak = float(max(np.abs(clean).max(), np.abs(noisy).max()))
        scale = min(1.0, 0.99 / max(peak, 1e-12))
        clean *= scale
        noisy *= scale
        role = str(clean_asset["oracle_role"])
        if role != str(noise_asset["oracle_role"]):
            raise ValueError("DNS clean/noise split roles do not match")
        file_id = f"v16_dns_{index:04d}"
        clean_path, noisy_path = _write_pair(
            output_root=output_root,
            role=role,
            file_id=file_id,
            clean=clean,
            noisy=noisy,
        )
        rows.append(
            {
                "oracle_role": role,
                "domain": "dns1_training",
                "file_id": file_id,
                "speaker_id": f"dns_reader_{clean_asset['reader_id']}",
                "noisy_path": _relative(noisy_path),
                "clean_path": _relative(clean_path),
                "sample_rate": SAMPLE_RATE,
                "num_samples": SAMPLES,
                "duration_seconds": DURATION_SECONDS,
                "target_snr_db": snr_db,
                "target_clean_rms_dbfs": target_clean_dbfs,
                "clean_source": clean_asset["path"],
                "noise_source": noise_asset["path"],
                "peak_scale": scale,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--select", action="store_true")
    action.add_argument("--materialize", action="store_true")
    parser.add_argument(
        "--voice-metadata",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata/train.csv"),
    )
    parser.add_argument(
        "--dns-root",
        type=Path,
        default=Path("data/external/dns1_interspeech2020"),
    )
    parser.add_argument(
        "--v15-exclusion",
        type=Path,
        default=Path("configs/v15/dns_cross_domain_dev_assets.json"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/v16/oracle_corpus_selection.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v16_oracle_corpus"),
    )
    parser.add_argument("--voice-train-items", type=int, default=1_200)
    parser.add_argument("--voice-calibration-items", type=int, default=200)
    parser.add_argument("--dns-train-items", type=int, default=240)
    parser.add_argument("--dns-calibration-items", type=int, default=60)
    parser.add_argument("--seed", type=int, default=16_010)
    args = parser.parse_args()

    voice_metadata = _resolve(args.voice_metadata)
    dns_root = _resolve(args.dns_root)
    exclusion_path = _resolve(args.v15_exclusion)
    selection_path = _resolve(args.selection)
    output_dir = _resolve(args.output_dir)
    if args.select:
        exclusion = json.loads(exclusion_path.read_text())
        voice = _select_voice(
            voice_metadata,
            training=args.voice_train_items,
            calibration=args.voice_calibration_items,
            seed=args.seed,
        )
        dns_clean, dns_noise = _select_dns(
            dns_root,
            exclusion,
            training=args.dns_train_items,
            calibration=args.dns_calibration_items,
            seed=args.seed + 1,
        )
        selection = {
            "status": "selected_before_dns_download_or_oracle_scoring",
            "selection_seed": args.seed,
            "voice_metadata": _relative(voice_metadata),
            "voice_metadata_sha256": _sha256(voice_metadata),
            "v15_dns_exclusion": _relative(exclusion_path),
            "v15_dns_exclusion_sha256": _sha256(exclusion_path),
            "source_commit": "70f19285c36cca4df2338f9248775ddc50980c6b",
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": DURATION_SECONDS,
            "voice_items": voice,
            "dns_clean_assets": dns_clean,
            "dns_noise_assets": dns_noise,
            "counts": {
                "voice_training": args.voice_train_items,
                "voice_calibration": args.voice_calibration_items,
                "dns_training": args.dns_train_items,
                "dns_calibration": args.dns_calibration_items,
            },
            "data_boundary": (
                "Training and training-calibration only; excludes every "
                "V15 cross-domain development asset and all test metadata."
            ),
        }
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": selection["status"],
                    "selection": _relative(selection_path),
                    "counts": selection["counts"],
                },
                indent=2,
            )
        )
        return

    selection = json.loads(selection_path.read_text())
    voice_rows = _materialize_voice(
        selection["voice_items"],
        output_root=output_dir,
        seed=int(selection["selection_seed"]),
    )
    dns_rows = _materialize_dns(
        selection["dns_clean_assets"],
        selection["dns_noise_assets"],
        dataset_root=dns_root,
        output_root=output_dir,
        seed=int(selection["selection_seed"]) + 1,
    )
    frame = pd.DataFrame([*voice_rows, *dns_rows])
    train = frame[frame["oracle_role"] == "training"].copy()
    calibration = frame[
        frame["oracle_role"] == "training_calibration"
    ].copy()
    train_path = output_dir / "train_base.csv"
    calibration_path = output_dir / "calibration_base.csv"
    train.to_csv(train_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    manifest = {
        "status": "materialized_unlabelled_oracle_corpus",
        "selection": _relative(selection_path),
        "selection_sha256": _sha256(selection_path),
        "train_metadata": _relative(train_path),
        "train_metadata_sha256": _sha256(train_path),
        "calibration_metadata": _relative(calibration_path),
        "calibration_metadata_sha256": _sha256(calibration_path),
        "items": {
            "training": int(len(train)),
            "training_calibration": int(len(calibration)),
        },
        "domains": {
            str(key): int(value)
            for key, value in frame["domain"].value_counts().items()
        },
        "external_test_used": False,
        "development_set_used": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
