# Exploratory distribution-free stopping results

## Coding

The replay implementation and native cache adapter are complete, but the local
checkout lacks coding candidate scores/correctness (`data.jsonl`) and candidate
lengths. Therefore **there is no new coding performance result**. Saved aggregate
tables are insufficient to evaluate a new stopping rule. See README for a command
that runs once those existing caches are copied over; regeneration is not needed
if the old caches are available.

## Alignment

We evaluated five generators, each on 100 Alpaca prompts with 960 cached
candidates. One frozen 50/50 prompt split (seed 30), eight shared response-order
permutations per prompt, six prices, and a common 512-sample cap were used.
The primary baseline selects **any integer N from 1 through 512** using training
profit only; test plots also show the complete fixed-N curve. Two prespecified
policies differ only in checkpoint schedule: doubling versus checking after
every response, both starting at four samples and both retaining the positive
third-largest-utility guard.

Both estimate utility from rewards using Bradley–Terry against the observed
prefix's empirical 0.99 quantile. They estimate next cost from observed mean
response length. No parametric utility/reward distribution is fitted. The
full-pool 0.99 quantile is used only for evaluating selected responses.

### Result: checking every response helps, but the current rule is not competitive

Mean test profits across the five generators, in dollars per task:

| Price per character | Train-selected fixed N | Doubling | Every response |
|---:|---:|---:|---:|
| 0.00000002 | 0.61446 | 0.60797 | 0.60417 |
| 0.00000010 | 0.54715 | 0.53588 | 0.53663 |
| 0.00000020 | 0.50257 | 0.46670 | 0.48537 |
| 0.00000100 | 0.39029 | 0.27853 | 0.34849 |
| 0.00000200 | 0.33854 | 0.19139 | 0.28169 |
| 0.00001000 | 0.21629 | -0.03442 | 0.09591 |

Across the 30 generator–price settings, doubling never improves on the
training-selected fixed count; checking every response improves in only two.
The unweighted mean relative profit differences are -33.55% and -15.29%,
respectively. These summaries span very different prices; the table shows the
important dependence on cost. They are exploratory results from one split,
not replacements for the paper's existing multiple-split experiments.

At the highest price, the doubling policy generates 56.23 responses on average
and the sequential policy 34.17; the selected fixed counts are only 4–6.
For Mistral specifically, fixed N=6 achieves profit 0.18919, doubling achieves
-0.08681 with 41.87 samples, and checking every response achieves 0.06311 with
26.31 samples. Extra quality does not offset the extra generation cost.

At the lowest price, costs are small and using most of the cap is favorable;
earlier stopping can instead reduce profit. Therefore the sequential schedule
is not uniformly better, even though it substantially reduces excessive
generation in the expensive regimes.

### Interpretation

The gain statistic can be implemented without a fitted distribution, and
one-by-one stopping is straightforward. However, this direct practical variant
is too reluctant to stop in the expensive regimes. The experiment does not
separate conservatism of the gain statistic from error in the unknown alignment
reference; both may matter. No coefficients or rules were retuned after seeing
these results. A natural future comparison would calibrate conservatism using
training data, and distinguish reference estimation from the stopping statistic,
but that would be a different algorithm and require a fresh held-out evaluation.

The theorem does not claim finite-price near-optimality here: DMRL is unverified,
utilities are prefix-dependent estimates, costs use a running sample mean, and
cached candidates are sampled without replacement. The results therefore do not
contradict the asymptotic fixed-cost theorem. They do show that the theorem's
algorithm should not be described as already successful on these tasks.

### Audit and reproducibility

- `results/<generator>/seed30/summary.csv` reports every fixed N and both rules.
- `comparisons.csv` gives per-setting absolute profit, quality, cost, mean sample
  count, and paired prompt-bootstrap profit-difference intervals. The intervals
  condition on training and are pointwise, not simultaneous over 30 settings.
- All five generators use the same held-out prompt split. They are not 250
  independent prompts; no pooled independence claim is made.
- Available lengths are exact text character counts, not model-token counts.
  Cost units are preserved explicitly; token-based results require token lengths.
- The paper and all legacy result tables were left unchanged.
- Unit tests cover formula/checkpoints, same-guard sequential decisions, ties,
  zeros, cap handling, unseen-prefix invariance, coding correctness isolation,
  held-out calibration, and native coding/alignment loaders.
- `mistral_initial_check/` preserves the first smoke run. Its sequential guard
  differed only at zero utilities (impossible for the finite clipped sigmoid
  used here). The final Mistral run was repeated from the frozen common source
  so the primary `results/` directory has consistent code provenance.
