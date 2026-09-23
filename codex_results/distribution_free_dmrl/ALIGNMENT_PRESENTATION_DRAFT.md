# Alignment figure and theory connection — proposed wording

These are staging notes under the experiment directory. No manuscript source
or manuscript figure has been replaced.

## Figure caption

**Alignment with FSFairX rewards.** (a) Relative profit improvement over a fixed
generation count selected on training prompts. (b) Generation-cost savings
relative to the cheapest retrospective mixture of fixed counts matching the
adaptive policy's attained mean BT utility. Each curve represents one generator.
Shading shows pointwise 95% prompt-bootstrap intervals from 1,000 resamples,
conditional on the selected policies and fixed-count baselines. Prompt weights
are shared across overlapping splits. Quality varies with price in (b).

The cost comparator is recomputed in each bootstrap resample. The intervals
are not simultaneous and do not account for development-time policy selection.
These qualifications can be stated in the evaluation protocol rather than
repeated inside the figure.

**Provenance for this draft figure:** existing caches charge recorded characters,
not tokenizer counts. The horizontal axis uses `10^6 p` without relabeling the
unit as tokens. A tokenizer-based regeneration requires replacing the numerical
results before a caption identifies p as a per-token price.

## Proposed connection from theory to the alignment algorithm

The DMRL analysis identifies two ingredients in the value of another response:
the probability of improving on the incumbent and the expected size of that
improvement. Its stopping rule estimates these quantities from upper order
statistics, without fitting a parametric utility distribution. We use the same
decomposition to construct an empirical alignment policy. From the responses
observed for the current prompt, we estimate upper-tail excesses in exponentiated
reward coordinates and translate them into Bradley–Terry utility increments.
The policy averages the estimated improvement scale across prefixes and compares
it with a variance-adjusted estimate of next-response cost. It makes this
comparison after every response and requires no separate training fit.

This wording describes a structural connection, not a transfer of the theorem.
The practical policy uses an estimated utility benchmark, a wider empirical
tail, every-response checks, smoothing, and empirical cost optimism. In
particular, its record-probability approximation and cost estimate are not
certified confidence bounds. The theorem assumes DMRL for observed utility;
the practical construction uses residuals of exponentiated rewards, for which
that assumption has not been established. Its performance must therefore be
evaluated experimentally.

## Suggested alignment-section flow for the next manuscript revision

1. Define BT utility and distinguish the full-pool evaluation reference from
   the online estimated benchmark; introduce exponentiated coordinates before
   using them in an algorithm.
2. Give the structural connection above and a short pseudo-code algorithm with
   explicit inputs and fixed parameters. Put the exact tail, smoothing, and
   cost-adjustment formulas in the appendix.
3. Specify the five generators, FSFairX/Mistral reward models, 100 prompts,
   eight replay orders, five overlapping splits, six prices, and cap 960.
   Explain that training prompts select fixed-N comparators, not the adaptive
   policy. Distinguish development-time configuration choice from deployment.
4. Present profit first: 5.55% for FSFairX and 4.998% for Mistral. The same
   configuration is used across generators and prices and was frozen before
   FSFairX evaluation. The two reward-model results share responses/prompts.
5. Present price-indexed matched-quality savings: 27.73% for FSFairX and 25.15%
   for Mistral. Explain that each point matches the quality attained at that
   price, rather than a common requested quality across the horizontal axis.
6. Keep retrospective target-quality frontiers separate. They do not establish
   a deployable training-free target controller.
7. Preserve the near-direct theoretical-policy comparison as an ablation. Its
   profit changes are -31.56%/ -36.86% for FSFairX/Mistral with doubling, and
   -13.26%/ -16.48% with every-response checks. These losses help explain why
   practical modifications are needed, but do not refute the asymptotic theorem.

Do not carry forward old ten-split, fitted-exponential-policy, matched-cost
win-rate, or trained-target-controller claims as results of this new policy.
