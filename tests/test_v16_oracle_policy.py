from __future__ import annotations

import pandas as pd

from scripts.generate_v16_oracle_labels import (
    _annotate_candidates,
    _select_candidate,
    _validate_metadata,
)


POLICY = {
    "feasibility": {
        "pesq": {
            "minimum_noisy_delta": -0.005,
            "minimum_full_strength_delta": -0.02,
        },
        "si_sdr": {"minimum_noisy_delta_db": -0.1},
        "stoi": {"minimum_noisy_delta": -0.001},
        "estoi": {"minimum_noisy_delta": -0.002},
    },
    "utility": {
        "metrics": {
            "pesq": {"weight": 0.30, "scale": 0.10},
            "si_sdr": {"weight": 0.15, "scale": 1.0},
            "stoi": {"weight": 0.35, "scale": 0.01},
            "estoi": {"weight": 0.20, "scale": 0.02},
        }
    },
    "missing_metric_policy": {
        "all_strengths_unavailable": (
            "exclude_and_renormalize_available_utility_weights"
        ),
        "partial_availability": "fail",
        "required_available_metrics": ["si_sdr", "stoi", "estoi"],
    },
}


def _row(strength, pesq, si_sdr, stoi, estoi):
    return {
        "strength": strength,
        "noisy_pesq": 1.0,
        "noisy_si_sdr": 0.0,
        "noisy_stoi": 0.8,
        "noisy_estoi": 0.7,
        "enhanced_pesq": pesq,
        "enhanced_si_sdr": si_sdr,
        "enhanced_stoi": stoi,
        "enhanced_estoi": estoi,
    }


def test_oracle_prefers_best_safe_utility() -> None:
    rows = [
        _row(0.0, 1.0, 0.0, 0.8, 0.7),
        _row(0.5, 1.1, 0.2, 0.82, 0.72),
        _row(1.0, 1.11, 0.3, 0.79, 0.69),
    ]
    selected, reason = _select_candidate(
        _annotate_candidates(rows, POLICY)
    )
    assert selected["strength"] == 0.5
    assert reason == "feasible_maximum_utility"


def test_oracle_uses_minimum_violation_fallback() -> None:
    rows = [
        _row(0.0, 0.8, -0.2, 0.79, 0.69),
        _row(0.5, 0.9, -0.3, 0.78, 0.68),
        _row(1.0, 1.0, -0.4, 0.77, 0.67),
    ]
    annotated = _annotate_candidates(rows, POLICY)
    selected, reason = _select_candidate(annotated)
    assert selected["strength"] == min(
        annotated,
        key=lambda row: row["constraint_violation_normalised"],
    )["strength"]
    assert reason == "fallback_minimum_constraint_violation"


def test_oracle_excludes_fully_unavailable_pesq() -> None:
    rows = [
        _row(0.0, None, 5.0, 0.37, 0.375),
        _row(0.5, None, 3.3, 0.36, 0.372),
        _row(1.0, None, -28.0, 0.10, 0.08),
    ]
    for row in rows:
        row["noisy_pesq"] = None
        row["noisy_si_sdr"] = 5.0
        row["noisy_stoi"] = 0.37
        row["noisy_estoi"] = 0.375
    annotated = _annotate_candidates(rows, POLICY)
    selected, reason = _select_candidate(annotated)
    assert selected["strength"] == 0.0
    assert selected["missing_metrics"] == "pesq"
    assert selected["pesq_floor"] is None
    assert reason == "feasible_maximum_utility"


def test_metadata_requires_explicit_training_role(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "file_id": "x",
                "speaker_id": "s",
                "noisy_path": "noisy.wav",
                "clean_path": "clean.wav",
                "sample_rate": 16_000,
                "num_samples": 16_000,
                "duration_seconds": 1.0,
                "oracle_role": "development",
            }
        ]
    )
    try:
        _validate_metadata(frame, tmp_path / "metadata.csv")
    except ValueError as error:
        assert "Unsupported oracle roles" in str(error)
    else:
        raise AssertionError("Development metadata was accepted")
