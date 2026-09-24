# Adaptive coding: isotonic calibration plus non-parametric DMRL

The implementation is [adaptive_coding.py](adaptive_coding.py). Its offline
profile contains an increasing isotonic map from CodeScaler reward to
`P(correct)` and output-length provenance. It deliberately contains no fitted
tail distribution.

Correctness labels are needed only on a disjoint calibration set to learn the
isotonic map and tune policy controls. The online policy receives only the raw
reward and paid output-token count for each completed generation. Final
correctness is revealed after stopping for evaluation.

## Adopted paper policy: top three, full history

The paper now uses `matched_top3_full_history`, saved in
`codex_results/code/coding_top3_adopted_20260924`. It computes the mean excess
of the two largest calibrated utilities above the third-largest, beginning
at response four. It averages **all** excess estimates from count four
onward, including zero-cutoff estimates. It retains the frozen gain multiplier,
cost adjustment, minimum, cap, and remembered-crossing behavior of the earlier
calibration-selected policy. The retained minima range from **61 to 256**;
this is not the separate <=16-start variant.

Use `AdaptiveCoding.from_paper_settings(...)` to select this exact rule.
The general constructor's defaults remain the historical top-five settings
for backward compatibility. `statistic_start=4` is essential: accumulating
top-three spacings from count three produces a different policy.

Across prices $1.00, $1.25, $1.50, $1.75, $2.00 per million tokens, the
adopted rule's relative profit changes over Fixed-N are +1.31%, +2.37%,
+2.20%, +0.57%, +0.39%. It was adopted after exploratory comparisons;
these are not independent confirmation or evidence of significant superiority.
The manuscript gives the remaining DMRL and practical-rule qualifications.

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
import json
from pathlib import Path
from algorithm.adaptive_coding import AdaptiveCoding, CodingProfile

price = 1.25e-6  # dollars per output token
artifact = Path("codex_results/code/coding_top3_adopted_20260924")
settings = json.loads((artifact / "selected_policies.json").read_text())
record = next(r for r in settings if r["split"] == 0 and abs(r["price"] - price) < 1e-15)
rule = record["rule"]
profile = CodingProfile.load(
    "codex_results/code/coding_prices1_2_step025_20260923/profiles/split_00.json"
)
policy = AdaptiveCoding.from_paper_settings(
    profile, price,
    multiplier=rule["alpha"], cost_adjustment=rule["beta"],
    minimum=rule["minimum"], cap=rule["cap"],
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

`price` is cost per output token. Profit is
`correct - price * total_output_tokens` and `total_cost` charges every observed
token. The returned response is the maximum raw CodeScaler reward. This
preserves the usual best-of-N ranking when isotonic calibration has a flat
probability region; calibrated probabilities drive stopping, not tie-breaking.

Use the profile paired with the saved split; this example reproduces one
setting, not deployment validation on new problems. Calls charge actual
output tokens. The named constructor does not fit or retune anything.

## Historical top-five rule and calibration

Starting at five observations, let `u[1] >= ... >= u[5]` be the five largest
calibrated success probabilities. The local non-parametric residual scale is

~~~text
m_n = mean(u[1:4] - u[5]).
~~~

The estimated next-sample improvement is `multiplier * smooth(m_n) / n`.
The legacy constructor's default smoother is the cumulative mean of observed residual scales;
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

This driver describes the historical parameter-selection study, not a fresh
tuning run for the adopted top-three rule. To regenerate the adopted table
from saved outcomes, run `coding_scripts/adopt_coding_top3.py` under
`codex_results/code` with `--source coding_top3_full_history_20260924` and a
new output directory. `check_adopted_coding_replay.py` checks the public API
against saved paired trajectories for all 50 adopted split-price settings.

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
