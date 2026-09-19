#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_utility_frontier_validation_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  codex_results/coding_utility_improvement/profiled_utility_frontier.py \
  --profile codex_results/code/coding_scripts/coding_target_tail_decay_profile.csv \
  --output-dir codex_results/coding_utility_improvement/results/frontier_validation \
  --profile-top-k 20 \
  --selection-profile-weight 0.5 \
  --selection-risk-penalty 0.1 \
  --split-start 65 \
  --splits 5 \
  --test-permutations 16 \
  --workers 8
