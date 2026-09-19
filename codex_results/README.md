# Reproducible experiment results

This directory is intentionally split into immutable experiment snapshots and
fresh outputs:

- `code/alignment_scripts`: alignment implementation, tests, and runner;
- `code/coding_scripts`: raw-reward BT/exponential and train-only isotonic
  probability-space coding implementations, tests, and runners;
- `results`: fresh outputs only.

Fresh runs completed in the requested order:

1. `results/alignment`;
2. `results/coding_utility_gap` (raw-reward baseline);
3. `results/coding_utility_gap_calibrated` (separate isotonic-calibrated
   Gaussian and exponential-tail comparison);
4. `results/coding_target_quality_calibrated` (single-policy target-quality
   evaluation with the two requested Fixed-N comparators).

The calibrated experiments fit isotonic `P(correct | raw reward)` on each
41-problem training half, freeze it for the corresponding 42-problem test
half, and fit both candidate reward distributions after transformation.
Independent bounded-support confirmations are under each calibrated result's
`improved` subfolder. `results/INVESTIGATION.md` records the successful
probability-support correction, paired development evidence, independent
confirmation results, and the target-quality limitation that remains.
`MANIFEST.sha256` records all secured source and final artifacts.

The later, isolated utility-maximization follow-up is in
`coding_utility_improvement`.  Its final `profiled_utility_frontier` result
keeps the shifted-exponential UCB-Pandora blueprint, freezes selection before
split IDs 75--84, and has positive mean utility gain at all ten reporting
divisors.  See `coding_utility_improvement/results/frontier_final/REPORT.md`.

The previous experiment tree was removed from this directory before these
runs. A temporary recoverable backup was placed at
`/tmp/codex_results_before_rebuild_20260803`.
