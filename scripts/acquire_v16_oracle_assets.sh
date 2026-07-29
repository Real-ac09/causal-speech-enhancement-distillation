#!/usr/bin/env bash
set -euo pipefail

project_root="/home/mohamedb/Documents/cn-vqg-speech-enhancement"
dataset_root="${project_root}/data/external/dns1_interspeech2020"
selection="${project_root}/configs/v16/oracle_corpus_selection.json"
archive="/tmp/cnvqg-git-lfs-linux-amd64-v3.7.1.tar.gz"
lfs_root="/tmp/git-lfs-3.7.1"
lfs_url="https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz"
lfs_sha256="1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08"

if [[ ! -x "${lfs_root}/git-lfs" ]]; then
    curl --fail --location "${lfs_url}" --output "${archive}"
    printf '%s  %s\n' "${lfs_sha256}" "${archive}" | sha256sum --check -
    tar --extract --gzip --file "${archive}" --directory /tmp
fi

include_paths="$(
    jq --raw-output \
        '[.dns_clean_assets[].path, .dns_noise_assets[].path] | join(",")' \
        "${selection}"
)"
PATH="${lfs_root}:${PATH}" git -C "${dataset_root}" lfs pull \
    --include="${include_paths}" --exclude=""

LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
    PYTHONPATH="${project_root}/src" \
    /home/mohamedb/miniconda3/envs/cnvqg/bin/python \
    "${project_root}/scripts/prepare_v16_oracle_corpus.py" \
    --selection "${selection}" \
    --output-dir "${project_root}/data/processed/v16_oracle_corpus" \
    --materialize
