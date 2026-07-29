#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/mohamedb/miniconda3/envs/cnvqg/bin/python"
project_root="/home/mohamedb/Documents/cn-vqg-speech-enhancement"
checkpoint="checkpoints/v15/preservation/quiet_level_seed1200/epoch_003.pt"
output_root="results/v15/preservation/quiet_level_seed1200"
cross_output="${output_root}/cross_domain_dev"
voice_output="${output_root}/voicebank_dev400"

cd "${project_root}"
export LD_LIBRARY_PATH="/home/mohamedb/miniconda3/envs/cnvqg/lib"
export PYTHONPATH="src"

if [[ ! -f "${cross_output}/summary.json" ]]; then
    "${python_bin}" scripts/evaluate.py \
        --checkpoint "${checkpoint}" \
        --metadata data/processed/dns_cross_domain_dev/metadata.csv \
        --output-dir "${cross_output}" \
        --device cuda \
        --weights model
fi

if [[ ! -f "${voice_output}/summary.json" ]]; then
    "${python_bin}" scripts/evaluate.py \
        --checkpoint "${checkpoint}" \
        --metadata \
        data/processed/voicebank_demand/metadata/v12_architecture_selection_400.csv \
        --output-dir "${voice_output}" \
        --device cuda \
        --weights model
fi

"${python_bin}" scripts/compare_evaluations.py \
    --reference \
    v14_2=results/v15/cross_domain_dev/baselines_seed1200/v14_2 \
    --candidate "quiet_level=${cross_output}" \
    --output-dir "${output_root}/cross_domain_comparison" \
    --bootstrap-samples 20000 \
    --seed 15012

"${python_bin}" scripts/compare_evaluations.py \
    --reference v14_2=results/v14/distillation/winner_dev400 \
    --candidate "quiet_level=${voice_output}" \
    --output-dir "${output_root}/voicebank_comparison" \
    --bootstrap-samples 20000 \
    --seed 15013

"${python_bin}" scripts/gate_v15_quiet_level.py \
    --gates configs/v15/promotion_gates.yaml \
    --cross-metadata data/processed/dns_cross_domain_dev/metadata.csv \
    --cross-reference \
    results/v15/cross_domain_dev/baselines_seed1200/v14_2 \
    --cross-candidate "${cross_output}" \
    --voice-reference results/v14/distillation/winner_dev400 \
    --voice-candidate "${voice_output}" \
    --output-dir "${output_root}/gate" \
    --bootstrap-samples 20000 \
    --bootstrap-seed 15014
