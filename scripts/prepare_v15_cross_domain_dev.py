#!/usr/bin/env python3
"""Select and deterministically mix a DNS1-training cross-domain dev set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

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


def _select_assets(
    dataset_root: Path,
    *,
    count: int,
    seed: int,
) -> dict[str, object]:
    clean_candidates: list[dict[str, object]] = []
    for path in sorted((dataset_root / "datasets" / "clean").glob("*.wav")):
        pointer = _pointer(path)
        reader = READER.search(path.name)
        if pointer is None or reader is None:
            continue
        oid, size = pointer
        if size < 320_000 or size > 8_000_000:
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
    clean_assets: list[dict[str, object]] = []
    seen_readers: set[str] = set()
    for item in clean_candidates:
        reader_id = str(item["reader_id"])
        if reader_id in seen_readers:
            continue
        seen_readers.add(reader_id)
        clean_assets.append(item)
        if len(clean_assets) == count:
            break

    noise_candidates: list[dict[str, object]] = []
    for path in sorted((dataset_root / "datasets" / "noise").glob("*.wav")):
        pointer = _pointer(path)
        if pointer is None:
            continue
        oid, size = pointer
        if size < 160_000 or size > 16_000_000:
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
    noise_assets = noise_candidates[:count]
    if len(clean_assets) != count or len(noise_assets) != count:
        raise RuntimeError(
            f"Could select only {len(clean_assets)} clean and "
            f"{len(noise_assets)} noise assets"
        )

    test_root = (
        dataset_root / "datasets" / "test_set" / "synthetic" / "no_reverb"
    )
    test_hashes = {
        _sha256(path)
        for directory in ("clean", "noisy")
        for path in (test_root / directory).glob("*.wav")
    }
    selected_oids = {
        str(item["lfs_oid_sha256"])
        for item in [*clean_assets, *noise_assets]
    }
    exact_test_hash_overlap = sorted(selected_oids & test_hashes)
    if exact_test_hash_overlap:
        raise RuntimeError("Selected DNS training assets overlap test files")

    return {
        "status": "assets_selected_before_download_or_mixing",
        "source_repository": "https://github.com/microsoft/DNS-Challenge.git",
        "source_branch": "interspeech2020/master",
        "source_commit": "70f19285c36cca4df2338f9248775ddc50980c6b",
        "selection_seed": seed,
        "items": count,
        "unique_clean_readers": count,
        "clean_assets": clean_assets,
        "noise_assets": noise_assets,
        "exact_test_hash_overlap": exact_test_hash_overlap,
        "test_files_excluded": 300,
        "mixing_plan": {
            "sample_rate": 16_000,
            "duration_seconds": 10.0,
            "snr_levels_db": list(SNR_LEVELS),
            "target_clean_rms_levels_dbfs": list(CLEAN_RMS_LEVELS),
            "peak_limit": 0.99,
        },
        "policy": (
            "Cross-domain development only. Never merge with DNS1 no-reverb "
            "test results or present as an external test."
        ),
    }


def _load_mono(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if rate != sample_rate:
        values = librosa.resample(
            values, orig_sr=rate, target_sr=sample_rate
        ).astype(np.float32)
    return values


def _clean_segment(values: np.ndarray, length: int, selector: int) -> np.ndarray:
    if len(values) < length:
        repeats = int(np.ceil(length / max(1, len(values))))
        return np.tile(values, repeats)[:length].copy()
    starts = np.arange(0, len(values) - length + 1, 16_000)
    if starts[-1] != len(values) - length:
        starts = np.append(starts, len(values) - length)
    rms = np.asarray(
        [
            np.sqrt(np.mean(values[start : start + length] ** 2) + 1e-12)
            for start in starts
        ]
    )
    top = starts[np.argsort(rms)[-min(3, len(starts)) :]]
    start = int(top[selector % len(top)])
    return values[start : start + length].copy()


def _noise_segment(values: np.ndarray, length: int, selector: int) -> np.ndarray:
    if len(values) < length:
        repeats = int(np.ceil(length / max(1, len(values))))
        values = np.tile(values, repeats)
    maximum = len(values) - length
    start = selector % (maximum + 1) if maximum > 0 else 0
    return values[start : start + length].copy()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2) + 1e-12))


def _materialize(
    dataset_root: Path,
    selection: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    sample_rate = 16_000
    length = sample_rate * 10
    clean_output = output_dir / "audio" / "clean"
    noisy_output = output_dir / "audio" / "noisy"
    clean_output.mkdir(parents=True, exist_ok=True)
    noisy_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    pair_hashes: list[dict[str, object]] = []

    for index, (clean_asset, noise_asset) in enumerate(
        zip(selection["clean_assets"], selection["noise_assets"])
    ):
        clean_source = dataset_root / str(clean_asset["path"])
        noise_source = dataset_root / str(noise_asset["path"])
        if _sha256(clean_source) != clean_asset["lfs_oid_sha256"]:
            raise ValueError(f"Clean LFS object mismatch: {clean_source}")
        if _sha256(noise_source) != noise_asset["lfs_oid_sha256"]:
            raise ValueError(f"Noise LFS object mismatch: {noise_source}")
        clean = _load_mono(clean_source, sample_rate)
        noise = _load_mono(noise_source, sample_rate)
        selector = int(_score(str(index), int(selection["selection_seed"]))[:12], 16)
        clean = _clean_segment(clean, length, selector)
        noise = _noise_segment(noise, length, selector // 7)

        target_clean_dbfs = CLEAN_RMS_LEVELS[
            (index // len(SNR_LEVELS)) % len(CLEAN_RMS_LEVELS)
        ]
        snr_db = SNR_LEVELS[index % len(SNR_LEVELS)]
        clean *= 10.0 ** (target_clean_dbfs / 20.0) / _rms(clean)
        target_noise_rms = _rms(clean) / (10.0 ** (snr_db / 20.0))
        noise *= target_noise_rms / _rms(noise)
        noisy = clean + noise
        peak = float(max(np.max(np.abs(noisy)), np.max(np.abs(clean))))
        peak_scale = min(1.0, 0.99 / max(peak, 1e-12))
        clean *= peak_scale
        noisy *= peak_scale

        file_id = f"dns_train_dev_{index:03d}"
        clean_path = clean_output / f"{file_id}.wav"
        noisy_path = noisy_output / f"{file_id}.wav"
        sf.write(clean_path, clean, sample_rate, subtype="PCM_16")
        sf.write(noisy_path, noisy, sample_rate, subtype="PCM_16")
        actual_noise = noisy - clean
        actual_snr = 20.0 * np.log10(_rms(clean) / _rms(actual_noise))
        rows.append(
            {
                "split": "cross_domain_development",
                "file_id": file_id,
                "speaker_id": f"dns_reader_{clean_asset['reader_id']}",
                "noisy_path": _relative(noisy_path),
                "clean_path": _relative(clean_path),
                "sample_rate": sample_rate,
                "num_samples": length,
                "duration_seconds": length / sample_rate,
                "target_snr_db": snr_db,
                "actual_snr_db": actual_snr,
                "target_clean_rms_dbfs": target_clean_dbfs,
                "actual_clean_rms_dbfs": 20.0 * np.log10(_rms(clean)),
                "clean_source": clean_asset["path"],
                "noise_source": noise_asset["path"],
                "peak_scale": peak_scale,
            }
        )
        pair_hashes.append(
            {
                "file_id": file_id,
                "clean_sha256": _sha256(clean_path),
                "noisy_sha256": _sha256(noisy_path),
            }
        )

    metadata_path = output_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    manifest = {
        "status": "frozen_cross_domain_development",
        "source_selection": selection,
        "metadata": _relative(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "items": len(rows),
        "duration_hours": len(rows) * 10.0 / 3600.0,
        "snr_item_counts": (
            pd.DataFrame(rows)["target_snr_db"].value_counts().sort_index().to_dict()
        ),
        "clean_level_item_counts": (
            pd.DataFrame(rows)["target_clean_rms_dbfs"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "pair_hashes": pair_hashes,
        "selection_role": (
            "V15 development and promotion screening; never an external test"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/external/dns1_interspeech2020"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("configs/v15/dns_cross_domain_dev_assets.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/dns_cross_domain_dev"),
    )
    parser.add_argument("--items", type=int, default=60)
    parser.add_argument("--seed", type=int, default=15_001)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    if args.select == args.materialize:
        parser.error("select exactly one of --select or --materialize")
    if args.select:
        selection = _select_assets(
            args.dataset_root, count=args.items, seed=args.seed
        )
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in selection.items()
                    if key not in {"clean_assets", "noise_assets"}
                },
                indent=2,
            )
        )
        return
    selection = json.loads(args.selection_output.read_text())
    manifest = _materialize(args.dataset_root, selection, args.output_dir)
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"source_selection", "pair_hashes"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
