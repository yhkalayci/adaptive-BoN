# Implementation alignment record

This record maps the simplified manuscript algorithms to the executable code.

## Shared contract

Both policies map reward and characters into a common utility scale, estimate
one-step improvement above the incumbent, use only already observed character
counts to predict the next cost, and stop when gain no longer covers cost.
Every reported cost is the exact cumulative output characters opened. Fixed-N
is never a cap, guard, fallback, or mixture inside either policy.

## Alignment

The selected policy is `local_exp_open5_conf06` in
`../code/alignment_scripts/alignment_simplification_study.py`; its streaming
operations are in `alignment_pandora_ucb.py`.

| Manuscript object | Implementation |
|---|---|
| Five initial responses | variant `min_open=5` passed to `pandora_stop_many` |
| Exponentiated score and inclusive upper half | `pandora_stop_many` heaps |
| Current-prompt shifted-exponential scale | `_exp_tail_ucb_ei_stats` with reward-prior weight zero |
| Confidence multiplier 0.6 and EI bonus 0.002 | selected `PandoraConfig` |
| BT benchmark and 48-point gain | `_exp_tail_ucb_ei_stats`, `_EXP_UNIT` |
| Running mean characters | `pandora_stop_many` with cost-prior weight zero |
| Full-pool 99th percentile used only for scoring | `make_trials` |

The placeholder `PriorFit` passed by the study is multiplied by zero in every
reward, benchmark, and cost equation. Focused tests replace it by extreme
values and verify invariant stopping decisions; the policy is therefore
genuinely training-free.

## Coding

The selected utility variant is `global_exp_q75_conf08_085x` in
`../code/coding_scripts/coding_simplification_study.py`. Core operations are in
`coding_ucb_three_objectives.py`.

| Manuscript object | Implementation |
|---|---|
| 83-problem cohort and 41/42 split | `load_problems`, `split_problems` |
| Train-only isotonic map | `fit_reward_space_isotonic` |
| Equal-problem global 75% location and scale | `fit_prior` |
| Exact global distribution online | `_ei_curve` with reward prior `Infinity` |
| Confidence 0.8 and bounded exponential quadrature | `_ei_curve` |
| Opportunity decay `3/n` | `tail_decay=1` in `pandora_stop_from_curve` |
| Running mean characters | cost-prior weight zero |
| Utility reservation divisor `0.85 D` | variant reservation rule `0.85x` |
| First-maximum tie behavior and exact cost | `_prefix_success`, evaluator |

The target experiment calls
`coding_target_quality_distribution_calibrated.py --simple-global-policy`.
`build_coding_target_simple_profile.py` averages held-out development accuracy
and log characters for the 19 reservation divisors. The final invocation sets
`--profile-train-accuracy-weight 0`, so the selector is only the inverse of
that frozen one-dimensional curve; no configuration grid or current-split
blend remains.
The runner also writes the exact held-out Fixed-N curves and constructs the
minimum-character common Fixed-N randomization whose aggregate accuracy equals
the adaptive aggregate accuracy. This randomization is an evaluation-only
oracle and never enters adaptive stopping.

## Baselines and audits

`build_revised_benchmarks.py` reads the confirmed simple-policy split rows,
reconstructs the exact test streams, verifies saved train-selected Fixed-N
utility within `2e-12`, and adds:

- test-oracle Fixed-N for both tasks;
- the alignment test-oracle prompt-specific character allocation;
- the coding train-selected fixed cumulative-character threshold;
- independent equal-character diagnostics; and
- train-selected Fixed-N target-transfer panels plus exact-quality Fixed-N
  diagnostics for alignment and coding.

`verify_claims.py` checks all displayed numbers against these results.
`verify_equations.py` independently implements the two displayed formulas and
checks their EI values and stops against the source implementations.

## Evaluation boundary

Alignment uses split IDs 30--39. Coding utility uses 75--84, the target profile
uses development IDs 55--59, and target confirmation uses 95--104. All coding
splits resample the same 83 problem identities; the manuscript states this
within-corpus limitation explicitly.
