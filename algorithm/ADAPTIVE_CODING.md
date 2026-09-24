# Adaptive coding: isotonic calibration plus non-parametric DMRL

The implementation is [adaptive_coding.py](adaptive_coding.py). Its offline
profile contains an increasing isotonic map from CodeScaler reward to
`P(correct)` and output-length provenance. It deliberately contains no fitted
tail distribution.

Correctness labels are needed only on a disjoint calibration set to learn the
isotonic map and tune policy controls. The online policy receives only the raw
reward and paid output-token count for each completed generation. Final
correctness is revealed after stopping for evaluation.

## Fit on calibration problems

~~~bash
python algorithm/adaptive_coding.py fit \
  --data /path/to/coding_correctness_rewards.jsonl \
  --output algorithm/profiles/coding_dmrl_profile.json \
  --seed 20260923
~~~

The JSONL has one record per problem and samples with `idx`, `r_score`,
`correct`, and `output_tokens`. Splitting is at the problem level to prevent
problem-specific leakage.

## Online use

~~~python
from algorithm.adaptive_coding import AdaptiveCoding, CodingProfile

profile = CodingProfile.load("algorithm/profiles/coding_dmrl_profile.json")
policy = AdaptiveCoding(
    profile,
    price=1 / 100_000,
    width=4,
    multiplier=1.0,
    smoothing="mean",
    cost_adjustment=2.0,
    cap=512,
)
programs = []

while True:
    program, output_tokens = generate_program(problem)
    reward = score_with_codescaler(problem, program)
    programs.append(program)
    decision = policy.observe(reward, output_tokens)
    if decision.should_stop:
        answer = programs[decision.best_index]
        break
~~~

`price` is cost per output token. Thus `price=1/D` implements profit
`correct - total_output_tokens/D` and `total_cost` charges every observed
token. The returned response is the maximum raw CodeScaler reward. This
preserves the usual best-of-N ranking when isotonic calibration has a flat
probability region; calibrated probabilities drive stopping, not tie-breaking.

## Top-four DMRL rule

Starting at five observations, let `u[1] >= ... >= u[5]` be the five largest
calibrated success probabilities. The local non-parametric residual scale is

~~~text
m_n = mean(u[1:4] - u[5]).
~~~

The estimated next-sample improvement is `multiplier * smooth(m_n) / n`.
The default smoother is the cumulative mean of observed residual scales;
`current` and `recent_half` are also available. The policy compares this gain
with running mean output-token cost. `cost_adjustment` can discount the cost
by its observed standard error; it changes no reward distribution assumption.
The policy stops when estimated gain is no greater than estimated next cost,
or at its frozen cap. If that comparison first favors stopping before the
calibration-tuned minimum, the signal is retained and the policy stops at the
minimum. This matches the batched experiment evaluator.

The experiment driver
`codex_results/code/coding_scripts/coding_token_profit_dmrl.py` tunes the
smoother, multiplier, cost adjustment, and a calibration-derived cap on the
offline problems only. It also tunes the Fixed-N comparator on that same
offline partition, freezes both, and evaluates paired unseen trajectories.
The optional `--tuning-cost-scale` multiplies token cost only when ranking
adaptive policy candidates on the calibration partition. Held-out profit
always uses the original output-token price.

## Audit

~~~bash
python algorithm/adaptive_coding.py audit \
  --profile algorithm/profiles/coding_dmrl_profile.json \
  --data /path/to/coding_correctness_rewards.jsonl \
  --partition holdout \
  --output /path/to/holdout_audit.json
~~~

The audit reports accuracy, profit, total generation counts, and total output
tokens. Correctness is consulted only after the policy has selected an index.
