#!/usr/bin/env python3
"""Create the dissertation research-freeze manifest exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("results/final_research_freeze/manifest.json")

ARTIFACTS = {
    "primary_checkpoint_seed1200": (
        "checkpoints/v14/distillation/mag_005/epoch_003.pt"
    ),
    "primary_checkpoint_seed1201": (
        "checkpoints/v14/distillation/"
        "mag_005_seed1201_fixed_epoch3/epoch_003.pt"
    ),
    "primary_checkpoint_seed1202": (
        "checkpoints/v14/distillation/"
        "mag_005_seed1202_fixed_epoch3/epoch_003.pt"
    ),
    "training_config_seed1200": (
        "configs/v14/generated/distillation/mag_005.yaml"
    ),
    "training_config_seed1201": (
        "configs/v14/distill_replication_seed1201.yaml"
    ),
    "training_config_seed1202": (
        "configs/v14/distill_replication_seed1202.yaml"
    ),
    "single_seed_frozen_protocol": "configs/v14/frozen_final_protocol.yaml",
    "replication_training_protocol": (
        "configs/v14/frozen_replication_protocol.yaml"
    ),
    "three_seed_evaluation_protocol": (
        "configs/v14/frozen_replication_evaluation.yaml"
    ),
    "replication_training_record": (
        "results/v14/replication/training_execution_record.json"
    ),
    "replication_evaluation_record": (
        "results/v14/replication_evaluation/execution_record.json"
    ),
    "standard_v13_aggregate": (
        "results/v14/replication_evaluation/standard/v13_aggregate.json"
    ),
    "standard_v14_2_aggregate": (
        "results/v14/replication_evaluation/standard/v14_2_aggregate.json"
    ),
    "standard_paired_comparison": (
        "results/v14/replication_evaluation/standard/"
        "v14_2_vs_v13_paired_hierarchical.json"
    ),
    "external_v13_aggregate": (
        "results/v14/replication_evaluation/external/v13_aggregate.json"
    ),
    "external_v14_2_aggregate": (
        "results/v14/replication_evaluation/external/v14_2_aggregate.json"
    ),
    "external_paired_comparison": (
        "results/v14/replication_evaluation/external/"
        "v14_2_vs_v13_paired_hierarchical.json"
    ),
    "runtime_seed1200": (
        "results/runtime/v14_2_distilled_seed1200_cpu.json"
    ),
    "voicebank_train_metadata": (
        "data/processed/voicebank_demand/metadata/train.csv"
    ),
    "voicebank_search_metadata": (
        "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
    ),
    "voicebank_locked_development_metadata": (
        "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
    ),
    "voicebank_standard_test_metadata": (
        "data/processed/voicebank_demand/metadata/test.csv"
    ),
    "dns1_external_metadata": (
        "data/processed/dns1_external/no_reverb_test.csv"
    ),
    "dns1_external_manifest": (
        "data/processed/dns1_external/manifest.json"
    ),
    "primary_model_source": (
        "src/cnvqg/models/predictive_noise_vq_mamba_v8.py"
    ),
    "native_streaming_source": "src/cnvqg/models/v8_native_streaming.py",
    "model_factory_source": "src/cnvqg/models/factory.py",
    "metric_source": "src/cnvqg/metrics/speech_metrics.py",
    "evaluation_source": "scripts/evaluate.py",
    "runtime_source": "scripts/measure_latency.py",
    "three_seed_report": "docs/results/v14_2_three_seed_external.md",
    "recipe7_conclusion": "results/v17/recipe7_conclusion.json",
    "recipe7a_checkpoint": (
        "checkpoints/v17/utility_safety_recipe7a_burnin_seed17047/"
        "epoch_004.pt"
    ),
    "recipe8_conclusion": "results/v18/recipe8_conclusion.json",
    "recipe8_selection": "results/v18/recipe8_selection/summary.json",
    "recipe8_checkpoint": (
        "checkpoints/v18/utility_safety_recipe8_two_stage_seed18000/"
        "epoch_007.pt"
    ),
    "recipe8_protocol": "configs/v18/frozen_recipe8_protocol.yaml",
    "freeze_creator": "scripts/create_final_research_freeze.py",
    "freeze_verifier": "scripts/verify_final_research_freeze.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing manifest after an intentional refreeze.",
    )
    args = parser.parse_args()
    output = ROOT / OUTPUT
    if output.exists() and not args.force:
        raise FileExistsError(
            f"{OUTPUT} already exists; use the verifier, not the creator"
        )
    missing = [
        path for path in ARTIFACTS.values() if not (ROOT / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Freeze artifacts are missing: {missing}")

    standard = load_json(
        ARTIFACTS["standard_v14_2_aggregate"]
    )["metrics"]
    external = load_json(
        ARTIFACTS["external_v14_2_aggregate"]
    )["metrics"]
    standard_delta = load_json(
        ARTIFACTS["standard_paired_comparison"]
    )["metrics"]
    external_delta = load_json(
        ARTIFACTS["external_paired_comparison"]
    )["metrics"]
    runtime = load_json(ARTIFACTS["runtime_seed1200"])
    manifest = {
        "schema_version": 1,
        "status": "research_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "writing_handoff_date": "2026-08-01",
        "git_head_at_freeze": git_head(),
        "primary_model": {
            "version": "V14.2",
            "research_label": "CN-VQG-GRU-T1-PD",
            "role": "final_deployable_dissertation_model",
            "architecture": "causal_temporal_core_v12",
            "parameters": 808095,
            "algorithmic_latency_ms": 20.0,
            "training_seeds": [1200, 1201, 1202],
            "fixed_epoch": 3,
            "canonical_deployment_checkpoint_role": (
                "primary_checkpoint_seed1200"
            ),
        },
        "claim_policy": {
            "architecture_reselection_permitted": False,
            "loss_reselection_permitted": False,
            "training_recipe_reselection_permitted": False,
            "epoch_or_checkpoint_reselection_permitted": False,
            "controller_promotion_permitted": False,
            "additional_test_evaluation_for_model_selection_permitted": False,
            "new_training_runs_are_part_of_final_claim": False,
            "voicebank_test_claim": "standard_comparability_not_pristine",
            "dns1_claim": "independent_external_evaluation_for_v14_2",
        },
        "frozen_claims": {
            "standard_voicebank_demand": {
                "items_per_seed": 824,
                "seeds": 3,
                "enhanced_pesq_mean": standard["enhanced_pesq"]["mean"],
                "enhanced_si_sdr_mean": standard["enhanced_si_sdr"]["mean"],
                "enhanced_stoi_mean": standard["enhanced_stoi"]["mean"],
                "enhanced_estoi_mean": standard["enhanced_estoi"]["mean"],
                "v14_2_minus_v13_pesq": (
                    standard_delta["enhanced_pesq"]["mean_delta"]
                ),
                "v14_2_minus_v13_stoi": (
                    standard_delta["enhanced_stoi"]["mean_delta"]
                ),
                "v14_2_minus_v13_estoi": (
                    standard_delta["enhanced_estoi"]["mean_delta"]
                ),
                "si_sdr_interpretation": "statistically_neutral",
            },
            "external_dns1": {
                "items_per_seed": 150,
                "seeds": 3,
                "enhanced_pesq_mean": external["enhanced_pesq"]["mean"],
                "enhanced_si_sdr_mean": external["enhanced_si_sdr"]["mean"],
                "enhanced_stoi_mean": external["enhanced_stoi"]["mean"],
                "enhanced_estoi_mean": external["enhanced_estoi"]["mean"],
                "v14_2_minus_v13_pesq": (
                    external_delta["enhanced_pesq"]["mean_delta"]
                ),
                "limitation": (
                    "mean_stoi_remains_below_unprocessed_noisy_input"
                ),
            },
            "deployment": {
                "cpu_streaming_rtf": runtime["streaming_rtf"],
                "frame_latency_p95_ms": runtime["frame_time_ms"]["p95"],
                "frame_latency_p99_ms": runtime["frame_time_ms"]["p99"],
                "persistent_state_mib": (
                    runtime["state_tensor_mebibytes_fp32"]
                ),
            },
        },
        "controller_study": {
            "role": "non_promoted_bounded_ablation",
            "best_balanced_candidate": "V17 Recipe 7a",
            "recipe8_decision": "stop_recipe8_and_report_two_stage_limit",
            "final_interpretation": (
                "Preservation routing improved violation severity but did not "
                "clear the frozen avoidable-violation gate."
            ),
        },
        "artifact_policy": {
            "paths_are_project_relative": True,
            "sha256_and_size_must_match": True,
            "manifest_regeneration_requires_explicit_force": True,
            "results_may_be_reformatted_but_not_recomputed_for_selection": True,
        },
        "artifacts": [
            {
                "role": role,
                "path": path,
                "size_bytes": (ROOT / path).stat().st_size,
                "sha256": sha256(ROOT / path),
            }
            for role, path in ARTIFACTS.items()
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    digest_path = output.with_suffix(".sha256")
    digest_path.write_text(f"{sha256(output)}  {OUTPUT}\n")
    print(f"Created {OUTPUT}")
    print(f"SHA-256: {sha256(output)}")


if __name__ == "__main__":
    main()
