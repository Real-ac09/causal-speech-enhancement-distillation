#!/usr/bin/env python3
"""Verify the final research-freeze manifest and frozen claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("results/final_research_freeze/manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(actual: float, expected: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > 1e-12:
        raise ValueError(f"Frozen claim changed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "research_frozen":
        raise ValueError("Manifest is not marked research_frozen")
    artifacts = manifest["artifacts"]
    paths = [item["path"] for item in artifacts]
    roles = [item["role"] for item in artifacts]
    if len(paths) != len(set(paths)) or len(roles) != len(set(roles)):
        raise ValueError("Duplicate artifact path or role")
    failures = []
    for item in artifacts:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            failures.append(f"size: {item['path']}")
            continue
        if sha256(path) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    if failures:
        raise ValueError("Freeze integrity failures:\n" + "\n".join(failures))

    by_role = {item["role"]: item["path"] for item in artifacts}
    standard = json.loads(
        (ROOT / by_role["standard_v14_2_aggregate"]).read_text()
    )["metrics"]
    external = json.loads(
        (ROOT / by_role["external_v14_2_aggregate"]).read_text()
    )["metrics"]
    standard_claim = manifest["frozen_claims"][
        "standard_voicebank_demand"
    ]
    external_claim = manifest["frozen_claims"]["external_dns1"]
    require_close(
        standard["enhanced_pesq"]["mean"],
        standard_claim["enhanced_pesq_mean"],
        "standard PESQ",
    )
    require_close(
        standard["enhanced_stoi"]["mean"],
        standard_claim["enhanced_stoi_mean"],
        "standard STOI",
    )
    require_close(
        external["enhanced_pesq"]["mean"],
        external_claim["enhanced_pesq_mean"],
        "external PESQ",
    )
    require_close(
        external["enhanced_stoi"]["mean"],
        external_claim["enhanced_stoi_mean"],
        "external STOI",
    )
    policies = manifest["claim_policy"]
    forbidden = (
        "architecture_reselection_permitted",
        "loss_reselection_permitted",
        "training_recipe_reselection_permitted",
        "epoch_or_checkpoint_reselection_permitted",
        "controller_promotion_permitted",
        "additional_test_evaluation_for_model_selection_permitted",
        "new_training_runs_are_part_of_final_claim",
    )
    if any(policies[name] for name in forbidden):
        raise ValueError("A forbidden post-freeze policy is enabled")
    print(
        "Research freeze verified: "
        f"{len(artifacts)} artifacts, all hashes and claims match."
    )


if __name__ == "__main__":
    main()
