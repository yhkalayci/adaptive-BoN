#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_simple_target_mpl"
mkdir -p "${MPLCONFIGDIR}"

DEVELOPMENT_DIR="codex_results/results/coding/simplification/target_development"
PROFILE="${DEVELOPMENT_DIR}/simple_policy_profile.csv"

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/coding_target_quality_distribution_calibrated.py" \
  --output-dir "${DEVELOPMENT_DIR}" \
  --simple-global-policy \
  --diagnostic-all-configs \
  --split-start 55 \
  --splits 5 \
  --train-permutations 4 \
  --test-permutations 8 \
  --folds 4 \
  --workers 8 \
  --seed 20260802

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/build_coding_target_simple_profile.py" \
  --diagnostic-csv "${DEVELOPMENT_DIR}/diagnostic_all_configs.csv" \
  --output "${PROFILE}"

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/coding_target_quality_distribution_calibrated.py" \
  --output-dir \
    codex_results/results/coding/simplification/target_final_profile_only \
  --simple-global-policy \
  --development-profile "${PROFILE}" \
  --profile-train-accuracy-weight 0 \
  --split-start 95 \
  --splits 10 \
  --train-permutations 16 \
  --test-permutations 48 \
  --folds 4 \
  --workers 8 \
  --seed 20260802
