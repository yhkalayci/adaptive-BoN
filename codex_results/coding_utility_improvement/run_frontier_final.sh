#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_utility_frontier_final_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  codex_results/coding_utility_improvement/profiled_utility_frontier.py \
  --profile codex_results/code/coding_scripts/coding_target_tail_decay_profile.csv \
  --frozen-selection \
    codex_results/coding_utility_improvement/results/frontier_validation/frozen_selection.csv \
  --output-dir codex_results/coding_utility_improvement/results/frontier_final \
  --split-start 75 \
  --splits 10 \
  --test-permutations 48 \
  --workers 8
