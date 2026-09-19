#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_utility_improvement_dev_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  codex_results/coding_utility_improvement/coding_utility_search.py \
  --output-dir codex_results/coding_utility_improvement/results/development \
  --grids exp_tail_current exp_tail_uncapped_decay \
  --splits 5 \
  --split-start 60 \
  --train-permutations 4 \
  --test-permutations 4 \
  --workers 8 \
  --diagnostic-all-configs
