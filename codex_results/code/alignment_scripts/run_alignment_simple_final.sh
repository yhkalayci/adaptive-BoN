#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_alignment_simple_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/alignment_simplification_study.py" \
  --output-dir \
    codex_results/results/alignment/simplification/final_confirmation \
  --variants reference_train_scale local_exp_open5_conf06 \
  --split-start 30 \
  --splits 10 \
  --train-permutations 4 \
  --test-permutations 8 \
  --workers 5 \
  --seed 20260802
