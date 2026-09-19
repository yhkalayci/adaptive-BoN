#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_utility_improvement_compare_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  codex_results/coding_utility_improvement/coding_utility_search.py \
  --output-dir codex_results/coding_utility_improvement/results/comparison \
  --grids exp_tail_current exp_tail_uncapped_decay \
  --splits 10 \
  --split-start 75 \
  --train-permutations 8 \
  --test-permutations 48 \
  --workers 8 \
  --development-profile \
    codex_results/coding_utility_improvement/results/development/development_profile.csv \
  --profile-weight 0.5 \
  --profile-risk-penalty 0.5
