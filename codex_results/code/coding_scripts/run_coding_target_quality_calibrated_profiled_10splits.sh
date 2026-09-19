#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_target_calibrated_profiled_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/coding_target_quality_distribution_calibrated.py" \
  --output-dir codex_results/results/coding/target_quality \
  --splits 10 \
  --split-start 95 \
  --train-permutations 16 \
  --test-permutations 48 \
  --folds 4 \
  --workers 8 \
  --seed 20260802 \
  --bounded-probability-models \
  --tail-decay-grid \
  --development-profile \
    "${SCRIPT_DIR}/coding_target_tail_decay_profile.csv"
