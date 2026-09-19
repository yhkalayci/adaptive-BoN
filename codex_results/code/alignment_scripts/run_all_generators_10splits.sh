#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_alignment_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python "${SCRIPT_DIR}/alignment_pandora_all_generators.py" \
  --data-dir dataset/alpaca \
  --reward-key mistral_rm_reward \
  --output-dir codex_results/results/alignment \
  --splits 10 \
  --split-start 30 \
  --train-permutations 4 \
  --test-permutations 8 \
  --workers 5 \
  --seed 20260802
