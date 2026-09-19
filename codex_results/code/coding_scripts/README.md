# Coding experiments

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
