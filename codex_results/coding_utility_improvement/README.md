# Coding utility improvement experiments

This directory is an isolated follow-up to the retained coding utility result
in `../results/coding/utility_gap`. It does not modify the retained scripts or
results.

The experiment compares two shifted-exponential policy grids under identical problem splits,
permutations, train-only isotonic reward calibration, exact output-character
cost, and Fixed-N baseline:

- `exp_tail_current`: the shifted-exponential portion of the retained bounded
  probability-space grid. It has no sample-count EI decay and may cap a
  trajectory at a multiple of train-tuned Fixed-N.
- `exp_tail_uncapped_decay`: shifted-exponential probability-space policies
  with no Fixed-N cap and tail-decay values `0, 0.25, 0.5, 1`.

Gaussian policies and the cross-family `train_selected_ucb`/`selected_ucb`
method are deliberately excluded. Hyperparameters are selected only within
the shifted-exponential family.

For each character divisor, each grid selects the configuration maximizing
cross-fitted training utility. A development profile can optionally stabilize
selection by blending current cross-fitted train utility with held-out utility
from designated development splits.

The selector frozen after splits 60--64 (profile construction) and 65--69
(selector validation) uses 50% current cross-fitted utility and 50% development
mean utility minus one half development split-to-split standard deviation. The
matched comparison must not be used to retune these weights.

The core reward models, expected-improvement calculation, stopping rule, and
data loading are imported from the retained implementation in
`../code/coding_scripts/coding_ucb_three_objectives.py`.

## Development run

Run a small set of designated development splits and evaluate every candidate:

```bash
bash coding_utility_improvement/run_development.sh
```

This writes `development_profile.csv`, which contains held-out utility means
and split-to-split standard deviations for every configuration and divisor.

## Matched comparison

Run the two exponential selectors together, optionally using
the frozen development profile:

```bash
bash coding_utility_improvement/run_comparison.sh
```

Outputs include per-split selections, utility summaries, configuration
selection frequencies, and a machine-readable method record.

## Stronger utility-frontier method

The capped-versus-decay comparison identified smooth tail decay as helpful,
but selecting a configuration only at the reporting divisor still left a
calibration bottleneck.  `profiled_utility_frontier.py` separates two roles:

- the **utility divisor** is the actual character price in the reported
  objective;
- the **reservation divisor** calibrates the expected-improvement threshold
  inside Pandora stopping.

This is exactly the shared blueprint with actual divisor `D` and

```text
A_n = (reservation_divisor / D) * (n / 3)^(-tail_decay),   B_n = 0.
```

Thus the implementation can use `EI * (n/3)^(-tail_decay) > chars /
reservation_divisor`, while the blueprint-equivalent expression remains
`A_n * EI > chars / D`.

This is not a change of objective and it is not a post-hoc test-set choice.
The shortlist comes from the frozen target-quality profile built on split IDs
55--59.  Split IDs 65--69 then choose one policy per utility divisor using a
50/50 blend of frozen-profile utility and arithmetic validation utility, with
a 0.1 validation-standard-deviation penalty.  Those meta-selection settings
were chosen by leave-one-validation-split-out checks.  The resulting ten
policies are frozen before split IDs 75--84 are evaluated.

Every retained candidate has all of the following properties:

- shifted-exponential upper-tail distribution, truncated to `[0, 1]`;
- strictly positive UCB confidence scale;
- train-only isotonic correctness calibration and distribution/cost priors;
- the same expected-improvement-versus-next-character-cost reservation test;
- no Fixed-N-derived cap (only the absolute 512-generation data horizon).

Run the disjoint validation and final stages in order:

```bash
bash codex_results/coding_utility_improvement/run_frontier_validation.sh
bash codex_results/coding_utility_improvement/run_frontier_final.sh
```

The validation artifacts are under `results/frontier_validation`; final
held-out artifacts are under `results/frontier_final`.
