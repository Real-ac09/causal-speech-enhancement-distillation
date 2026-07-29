#!/usr/bin/env bash
set -euo pipefail

# Acquire only the paired, synthetic, no-reverberation test subset from the
# official Microsoft DNS Challenge 1 repository. Git LFS is installed locally
# in /tmp; no system files are changed.

project_root="/home/mohamedb/Documents/cn-vqg-speech-enhancement"
dataset_root="${project_root}/data/external/dns1_interspeech2020"
archive="/tmp/cnvqg-git-lfs-linux-amd64-v3.7.1.tar.gz"
lfs_root="/tmp/git-lfs-3.7.1"
lfs_url="https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz"
lfs_sha256="1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08"
dns_repository="https://github.com/microsoft/DNS-Challenge.git"
dns_branch="interspeech2020/master"
include_path="datasets/test_set/synthetic/no_reverb/**"

if [[ ! -x "${lfs_root}/git-lfs" ]]; then
    curl --fail --location "${lfs_url}" --output "${archive}"
    printf '%s  %s\n' "${lfs_sha256}" "${archive}" | sha256sum --check -
    tar --extract --gzip --file "${archive}" --directory /tmp
fi

if [[ ! -d "${dataset_root}/.git" ]]; then
    if [[ -e "${dataset_root}" ]]; then
        echo "Refusing to overwrite non-repository path: ${dataset_root}" >&2
        exit 1
    fi
    GIT_LFS_SKIP_SMUDGE=1 PATH="${lfs_root}:${PATH}" \
        git clone --depth 1 --single-branch --branch "${dns_branch}" \
        "${dns_repository}" "${dataset_root}"
fi

PATH="${lfs_root}:${PATH}" git -C "${dataset_root}" lfs install --local
PATH="${lfs_root}:${PATH}" git -C "${dataset_root}" lfs pull \
    --include="${include_path}" --exclude=""

LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
    /home/mohamedb/miniconda3/envs/cnvqg/bin/python \
    "${project_root}/scripts/prepare_dns1_external_holdout.py" \
    --dataset-root "${dataset_root}" \
    --output-dir "${project_root}/data/processed/dns1_external"
