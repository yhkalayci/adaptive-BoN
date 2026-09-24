# Coding experiments

## Current non-parametric DMRL coding study

`coding_token_profit_dmrl.py` is the current token-cost analysis. A disjoint
calibration half fits only an increasing isotonic reward-to-correctness map.
Online stopping uses the mean excess of the top four calibrated utilities
above the fifth-largest utility, with no parametric tail fit. The script tunes
this policy and Fixed-N offline, freezes both, then evaluates paired unseen
problems with actual correctness and cumulative output-token cost.

```bash
python codex_results/code/coding_scripts/coding_token_profit_dmrl.py \
  --data /path/to/coding_correctness_rewards.jsonl \
  --output /path/to/results
```

The result directory includes per-split policies and trials, total generation
and token counts, split- and unique-problem-clustered intervals, CSV summaries,
an SVG plot, and a Markdown report.

### Optional controls and post-processing

The driver accepts `--tuning-cost-scale` (default `1.0`) and `--width`
(default `4`). A cost scale of `1.25` ranks the same adaptive policy candidates
with 25% more weight on calibration output-token cost. It leaves the online
stopping formula and the held-out profit calculation unchanged.

```bash
python codex_results/code/coding_scripts/coding_token_profit_dmrl.py \
  --data /path/to/coding_correctness_rewards.jsonl \
  --output /path/to/new_results \
  --divisors 600000,650000,700000,750000,1000000 \
  --outer-seed 20261023 --tuning-cost-scale 1.25
```

`coding_fixed_frontier.py` can evaluate a denser Fixed-N quality/token
frontier on exactly the same held-out split and permutation seeds without
retuning the adaptive policy.
`plot_coding_cost_regime_curves.py` and `plot_coding_followup_curves.py`
generate separate matched-quality cost-saving and profit-improvement SVGs
from saved runs. Generated datasets and evaluation results are not part of
the implementation.

## Historical manuscript policy

The manuscript uses one calibrated coding stopping family. Each 41-problem
training half fits raw reward to `P(correct)` by isotonic regression and then
fits one global shifted-exponential upper-quarter tail. Online stopping uses
confidence multiplier 0.8, running-mean character cost, opportunity decay
`3/n`, and no Fixed-N guard or cap.

## Utility

The one development-selected utility rule uses `D_res = 0.85 D` for every
economic divisor. Reproduce its final split IDs 75--84 with:

```bash
bash codex_results/code/coding_scripts/run_coding_simple_utility_final.sh
```

Output:

```text
codex_results/results/coding/simplification/final_confirmation_reservation/
```

## Target accuracy

The target selector freezes a development accuracy/cost curve for the same
single stopping family, then takes the cheapest reservation divisor reaching
the requested accuracy. It uses no current-split blend, safety margin,
interpolation, policy mixture, or configuration search. Reproduce the
development profile and final split IDs 95--104 with:

```bash
bash codex_results/code/coding_scripts/run_coding_simple_target_final.sh
```

Output:

```text
codex_results/results/coding/simplification/target_development/
codex_results/results/coding/simplification/target_final_profile_only/
```

The older profiled grid runners remain in this directory as historical
comparators used by the simplification study; they are not the manuscript's
selected policy.

## Response-character distribution check

```bash
bash codex_results/code/coding_scripts/run_response_char_distribution_check.sh
```

This writes the Alpaca, HH-RLHF, and Coding 1-by-3 mean-character histogram to
`codex_results/results/distribution_check/`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 /home1/kalayci/env/bin/python \
  -m unittest discover -s codex_results/code/coding_scripts -p 'test_*.py'
```
