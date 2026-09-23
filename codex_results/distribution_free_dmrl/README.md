# Distribution-free stopping: cached-response replay

## Current alignment policy

Use the [standalone online implementation](../../algorithm/adaptive_alignment.py)
and [algorithm guide](../../algorithm/ADAPTIVE_ALIGNMENT.md), with
[server instructions](../../SERVER_HANDOFF.md) for both reward models,
token-count inputs, and new-run plotting. The primary replay runner is
two_reward_models.py with method mean_costse2.
The records below describe the earlier development experiments, including
negative findings; they are retained alongside the current results.

The original experiment below evaluates the largest-three-observations idea
without fitting a utility distribution. Its policies differ from the final
smoothed alignment rule. No responses are generated and no GPU is used.

The latest **training-free** retry is summarized in
[`TRAINING_FREE_RETRY.md`](TRAINING_FREE_RETRY.md). It tests 24 additional
fixed policies, records negative results, and distinguishes savings at attained
quality from the harder task of controlling a requested quality without training.

The subsequent train-selected nonparametric study is documented separately in
[`nonparametric_followup/REPORT.md`](nonparametric_followup/REPORT.md). It tests
order-statistic spacings, recent record improvements, and remaining utility
range, with a cost-only length-budget control. The original results below are
preserved unchanged.

The next refinement adds wider order statistics, an optional training-only
quantile-bias correction, and target-quality evaluation. See
[`order_refinement_results/REPORT.md`](order_refinement_results/REPORT.md) for
profit, attained quality, matched-quality savings, and deployment requirements.

A further paired test of incumbent-aware tail-moment estimates is in
[`incumbent_aware_results/REPORT.md`](incumbent_aware_results/REPORT.md). It
preserves both negative findings and the very small overall improvement from
training-selecting among the added families. Independent audit outputs verify
the profit selections and optimal quality-matched comparators.

## Coding

The runner supports the existing `algorithm/bestofn_coding/data.jsonl` schema and
an NPZ length cache with `ids`, `indices`, and `chars` (or `tokens`). These coding
candidate caches are **not present in this checkout**. Aggregate result tables
cannot support replay of a new stopping policy; no new coding result is claimed.
Once those existing caches are available, run from the repository root:

```sh
python codex_results/distribution_free_dmrl/evaluate.py \
  --task coding --data algorithm/bestofn_coding/data.jsonl \
  --length-cache /path/to/coding_lengths.npz --length-key chars \
  --length-unit characters --prices 1e-5 5e-6 3.333333333e-6 2.5e-6 2e-6 1.666666667e-6 1.428571429e-6 1.25e-6 1.111111111e-6 1e-6 \
  --seed 75 --permutations 48 \
  --output codex_results/distribution_free_dmrl/results/coding_seed75
```

Only problems with at least one correct cached candidate are retained, matching
the existing cohort definition. On training problems only, direct isotonic
regression fits reward to success probability; the map is then frozen.
Stopping sees calibrated rewards, never correctness. Selection chooses the
largest raw reward (also a maximizer of the monotone calibration); this avoids
arbitrary selection changes inside calibration plateaus. Test profit uses
actual correctness minus paid generation cost, not predicted correctness.
Isotonic plateaus can make all three largest utilities identical, yielding zero
estimated gain and an immediate stop. DMRL is not established for these discrete
calibrated utilities, so this is an empirical concern, not a covered guarantee.

## Alignment

Five Alpaca response caches are available under `dataset/alpaca/`, each with 100
prompts and 960 candidates. The default reward field is `mistral_rm_reward`.
Example:

```sh
python codex_results/distribution_free_dmrl/evaluate.py \
  --task alignment --data dataset/alpaca/mistral_7b_output.merged_rm.jsonl.gz \
  --length-unit characters --prices 2e-8 1e-7 2e-7 1e-6 2e-6 1e-5 \
  --seed 30 --permutations 8 --cap 512 \
  --output codex_results/distribution_free_dmrl/results/mistral_7b/seed30
```

For stopping, the 0.99 reward quantile is estimated from the **observed prefix**
using the same order-statistic convention as the saved alignment runner. Current
prefix rewards become `sigmoid(reward - prefix_q99)`. Thus no parametric law is
fitted, but the benchmark must still be estimated. It is not fed in from the
unobserved candidate pool. Evaluation uses `sigmoid(selected_reward - full_q99)`
against the full saved pool's empirical reference, as in the original study.
The full reference is used only after stopping. The empirical reference is noisy
for small prefixes and all past utility estimates change as the prefix grows;
this is a pragmatic extension, not the theorem's fixed utility observation model.

## Policies and comparisons

- `dmrl_doubling`: checks at 4, 8, 16, ... paid samples. Let `low` be the
  third-largest observed utility; the estimated gain is
  `2 * (largest + second_largest - 2 * low) / n`. Stop when `low > 0` and
  gain is no greater than estimated next cost. This is the paper's exact
  parameter-free gain statistic and checkpoint schedule, with a common finite
  cap and estimated random cost substituted for known fixed cost.
- `dmrl_sequential`: the same statistic and positivity guard, checked after every
  sample starting at four. This prespecified pragmatic variant can react between
  doubling checkpoints. The checkpoint schedule is the only difference between
  the two policies. No theoretical guarantee is claimed for this variant.
- `fixed_N`: every integer `N=1,...,cap`, including 1–3 so the four-sample start
  is not artificially excused. The primary comparator chooses one N per price
  by mean **training** profit. The test-best N is a separately labeled oracle
  diagnostic, not an implementable selection rule.

All policies use exactly the same random orders within each prompt. Sampling
is without replacement from its saved pool. The common default cap is 512;
the runner returns the best opened response at the cap, including when the cap
is not a checkpoint. Every opened response is charged. Estimated next cost is
price times the observed prefix's mean response length. True cost is price
times its cumulative length. Utility is valued at at most one dollar.

The currently available alignment caches record response texts, not token
counts. These replay results therefore charge **characters**, accurately
identified in metadata. Token replays require recorded token-count fields:
pass `--length-key token_count --length-unit tokens` for alignment, or
`--length-key tokens --length-unit tokens` with a coding token cache. No automatic
conversion or tokenizer claim is made.

Split seed fixes a 50/50 split of prompts; no variant or threshold is chosen
using test outcomes. Per-prompt results average replay orders before uncertainty
calculation. The comparison interval is a paired bootstrap of held-out prompts,
conditional on the training split; replay orders are not treated as independent
prompts. Intervals are pointwise, not corrected for multiple prices/models.
Repeated splits would overlap and cannot be treated as independent datasets.

There is no transfer of the paper's theorem to these runs: costs are estimated,
the candidate pools are finite without-replacement samples, DMRL is unverified,
and alignment estimates its utility reference online. Negative results must
remain visible; the theory's asymptotic guarantee does not imply practical
near-optimality at these prices.

## Outputs and tests

Each run refuses to overwrite an existing output directory and creates:

- `METHOD.json`: split IDs, cost units, prices, cap, seed, policy source hash,
  and calibration knots when coding data are used.
- `summary.csv`: train/test mean quality, cost, samples, and profit for every N
  and both adaptive policies.
- `prompt_metrics.csv`: per-prompt averages for powers-of-two N, the cap,
  training-selected N, and adaptive policies.
- `comparisons.csv`: held-out adaptive results versus training-selected N,
  paired profit-difference intervals, and a test-oracle diagnostic.

Dependencies: Python 3.10+, NumPy; scikit-learn only for coding calibration.

```sh
python -m unittest discover -s codex_results/distribution_free_dmrl -v
python codex_results/distribution_free_dmrl/evaluate.py --help
```

Tests cover ties, all-zero utilities, cap exhaustion, exact gain/checkpoints,
the known-fixed-cost mode, raw-reward tie breaking, correctness-label isolation,
and invariance to unseen future rewards/costs. None of these tests is an LLM
performance result.
