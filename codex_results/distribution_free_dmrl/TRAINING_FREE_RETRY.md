# Training-free retry: performance and connection to theory

## Conclusion

This retry tested 24 additional fixed policies without offline training or a
fitted response distribution. It did **not** achieve the requested averages of
10% profit improvement and 35% matched-quality cost savings. Neither new family
improved on the previous primary training-free odds rule.

The best-supported existing choice remains `rank_jensen_bound`: **3.85% average
profit improvement and 24.37% average generation-cost savings at its attained
quality**. These are exploratory alignment results, not new coding results and
not a guarantee of hitting a user-specified quality target.

| Fixed policy | Mean profit change | Mean attained-quality cost saving |
|---|---:|---:|
| Previous primary: online odds/residual estimate | +3.85% | 24.37% |
| New primary: wider order-statistic spacing | +2.59% | 14.13% |
| New primary: raw-reward DMRL/Lipschitz bound | +1.34% | 8.99% |
| Closest statistic/schedule to the manuscript, with practical plug-ins | −36.86% | −11.13% |

The last row uses the largest-three statistic, multiplier four, and doubling
checkpoints. Unlike the theorem it estimates both the BT benchmark and random
mean cost from the current prefix, and imposes a finite cache cap. Its poor
results do not contradict the theorem, whose assumptions and asymptotic regime
are different.

The best spacing sensitivity variant *after inspecting this dataset* was
`quarter32_a1_sequential` (+2.85%, 14.79%). It was not the predeclared primary,
and selecting it retrospectively does not create independent confirmation.
All other variants, including negative results, remain in the result tables.

## What was tested

### Wider order statistics

After n samples, order the current utility estimates as v1 >= ... >= vn.
For a chosen tail width k, estimate mean residual life by

    m_hat = mean(v1,...,vk) - v(k+1).

Stop when `a*m_hat/n <= price*mean_observed_length`. The primary uses
`k=min(32,ceil(sqrt(n)))`, multiplier `a=1`, and every-response checks from
four samples. The sensitivity grid uses three widths, multipliers 1/2/4, and
every-response/doubling schedules: 18 policies, with no data-based adaptive
selection. This preserves the manuscript's order-statistic structure while
testing the practical cost of optimism and coarse checkpoints. The primary's
different width, multiplier, and schedule are not covered by the theorem.

### A bound that avoids estimating the BT benchmark

The second family uses raw rewards rather than estimated utilities. At a tail
threshold xi, estimate its tail probability p and mean residual m, and compute

    estimated gain = p*m*exp(-(current_max-xi)/m)/4.

Compare this with the same running estimate of next-call cost. There are three
tail widths, each including or excluding the current maximum in its residual
mean: six fixed variants. The primary uses a square-root tail width and includes
the maximum. No utility benchmark enters the stopping decision.

The population bound follows from DMRL and the 1/4-Lipschitz property of the BT
map. The short derivation and the remaining empirical-estimation gaps are in
[the theory connection note](TRAINING_FREE_THEORY_CONNECTION.md). It is a
genuine structural link, but not an end-to-end guarantee for this implementation.
The wider half-sample versions performed particularly poorly; excluding the
record was also harmful. Simpler population upper bounds did not translate
into better finite-sample decisions here.

## Comparison protocol

All runs reuse the same five Alpaca generator caches, 100 prompts per generator,
960 responses per prompt, eight cached replay orders, and six actual prices.
Seeds and data/source hashes are recorded in each `METHOD.json`. The adaptive
policies receive only the current response prefix and deployment price: there
is no trained predictor, calibration table, fitted stopping price, or offline
policy selection. Every generated response is paid for.

Profit is compared against fixed N selected using training prompts from all
integers 1 through 960. The adaptive policies do not use these training prompts;
the split is retained only for comparison. Results average over five overlapping
50/50 splits and then generator-price conditions. These are not 150 independent
experiments. The previous primary remains +3.16% against the stronger diagnostic
that selects the single best fixed N on test outcomes. Its average profit gain
is positive in 24/30 generator-price conditions; matched savings are positive
in 30/30.

For each actual-price adaptive policy, quality is evaluated using the full-cache
BT reference **only after the stopping decision**. We then compare its mean cost
with the cheapest retrospective mixture of fixed counts having that same mean
quality. This is a strong matched-quality comparator, not a claim that the
adaptive policy can attain an arbitrary supplied target without calibration.

Separately, `frontier_diagnostic.csv` reports requested qualities 0.30 to 0.60.
Both adaptive-price mixtures and fixed-count mixtures are selected retrospectively
for this diagnostic. The new spacing primary averages 16.52% savings over its
170/175 feasible split-target cases; the raw-reward primary averages 8.23% over
170/175. Infeasible targets are retained and flagged. These frontiers must not
be described as deployable training-free target controllers.

In particular, the previous odds rule's **direct** target controller was not
successful: it averaged −35.03% matched savings (more cost). The 24.37% figure
above describes its profit policies at their attained qualities; it does not
replace that negative target-control result. Nor is it directly comparable to
the older trained policy's approximately 20.7% average over seven target levels.

## Verification and scope

- All 46 experiment unit tests pass, including prefix measurability, future-cost
  isolation, exact first crossings, mandatory caps, and evaluation-label isolation.
- The audit scripts independently reconstruct profit accounting, confirm identical
  replay baselines, and check primary quality-matched mixtures with linear
  programming rather than the runner's convex-hull routine.
- `AUDIT.json` in each new result directory records the completed checks and
  shared-prompt bootstrap sensitivity. These conditional intervals are not
  independent confirmation after repeated development on these prompts.
- The previous odds primary has 150 independent LP checks recorded in
  `training_free_odds_results/PROFIT_MATCHED_AUDIT.json`.
- No new responses, GPU work, manuscript changes, or coding performance claims.
  Coding response-level caches remain unavailable. The replay cost unit remains
  the recorded character length, explicitly identified in metadata.

Results and runnable implementations:

- [Spacing runner](training_free_spacing.py), [results](training_free_spacing_results/profit.csv).
- [Raw-reward runner](training_free_lipschitz.py), [results](training_free_lipschitz_results/profit.csv).
- [Previous primary's matched-quality audit](training_free_odds_results/PROFIT_MATCHED_AUDIT.json).

The next research question is how to obtain a less conservative, statistically
controlled online improvement estimate—not how to select a favorable multiplier
from these test results. The present retry is evidence of a practical gap between
the available proof and the desired performance, not evidence that the numerical
targets are impossible.
