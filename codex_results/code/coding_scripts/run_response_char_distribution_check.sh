#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="/tmp/codex_response_char_distribution_mpl"
mkdir -p "${MPLCONFIGDIR}"

/home1/kalayci/env/bin/python \
  "${SCRIPT_DIR}/plot_response_char_distributions.py" \
  --output-dir codex_results/results/distribution_check
