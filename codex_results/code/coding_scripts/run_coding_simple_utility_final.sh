#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_coding_simple_utility_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/coding_simplification_study.py" \
  --output-dir \
    codex_results/results/coding/simplification/final_confirmation_reservation \
  --variants global_exp_q75_conf08_085x \
  --split-start 75 \
  --splits 10 \
  --test-permutations 48 \
  --workers 8 \
  --seed 20260802
