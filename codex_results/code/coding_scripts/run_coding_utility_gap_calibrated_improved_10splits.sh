#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_utility_calibrated_improved_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python "${SCRIPT_DIR}/coding_ucb_three_objectives.py" \
  --output-dir codex_results/results/coding/utility_gap \
  --splits 10 \
  --split-start 75 \
  --train-permutations 4 \
  --test-permutations 48 \
  --workers 8 \
  --seed 20260802 \
  --calibrated-reward-space \
  --bounded-probability-models \
  --skip-target-accuracy
