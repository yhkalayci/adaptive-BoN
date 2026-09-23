# Nonparametric stopping follow-up

## Coding

No new coding results are available: the required candidate-level cache is
absent. The previous coding replay adapter remains available. No new responses
were generated, and no GPU or manuscript changes were needed for this study.

## Alignment: positive results versus fixed N, with an important control

**Training-selected order-statistic stopping improves profit by 2.50% on
average versus training-selected fixed N.** A training-selected choice across
three nonparametric families improves it by 4.63%. However, a simple
training-selected response-length budget improves it by 6.00%, outperforming
both. Thus these experiments support adaptation to variable generation lengths;
they do not establish an additional benefit of these reward-based rules over
cost-only adaptation.

| Policy | Mean relative profit difference vs fixed N | Positive generator–price means |
|---|---:|---:|
| Original every-response top-three rule | −15.71% | 0/30 |
| Training-selected order-statistic spacing | +2.50% | 21/30 |
| Training-selected recent record improvements | −10.45% | 6/30 |
| Training-selected remaining utility range | +4.71% | 29/30 |
| Training-selected choice across all three families | +4.63% | 28/30 |

These family-specific results are diagnostics, not a test-selected combined
winner. The implementable combined comparison selects its family and parameters
using training profit only. The length-budget control is evaluated separately.

Mean relative differences by price, averaged over five generators and five
splits (percent):

| Price per recorded character | Order statistics | Training-selected combined | Length-budget control |
|---:|---:|---:|---:|
| 2e-8 | −0.12 | +0.03 | +0.01 |
| 1e-7 | −0.40 | +1.90 | +1.93 |
| 2e-7 | +1.16 | +4.26 | +4.18 |
| 1e-6 | +4.17 | +5.78 | +7.82 |
| 2e-6 | +4.56 | +7.52 | +9.26 |
| 1e-5 | +5.62 | +8.31 | +12.77 |

The spacing family's mean relative gains, aggregated across prices and splits,
are positive for every generator: Gemma 1.46%, Llama-3.1 3.46%, Llama-3.2 3.70%,
Mistral 0.56%, and Qwen 3.31%. Nevertheless, its overall profit is 3.21% below
the training-selected length-budget control; the combined method is 1.22% below
that control. These are means of within-setting relative differences, not
relative differences of pooled profits.

## What the algorithms do

All policies inspect every response, retain the highest-reward response, and
pay for every generated response. No parametric distribution is fitted.
At sample count n, alignment scores are transformed by Bradley–Terry against
the observed prefix's empirical 0.99 quantile. Estimated next cost is price
times observed mean response length. The full-pool reference is evaluation-only.

### 1. Order-statistic spacing

Let the largest observed utilities be v1 >= v2 >= ... . Estimate improvement as

    gain = multiplier / n * mean(v1 - v[k+1], ..., vk - v[k+1]).

Stop when this estimate is no greater than estimated next cost. Training chooses
k in {1, 2, 4} and multiplier in {0.125, 0.25, 0.5, 1, 2, 4}; minimum sample count
is max(4, k+1). The original rule is k=2 and multiplier=4. For k=1, the statistic
is the largest gap divided by n, the empirical leave-one-out decrease in the
sample maximum before multiplying by the chosen coefficient. This is not a
conditional unbiased estimate of the next improvement.

Training most often chooses k=4 and multiplier=1 (84/150 split–price–generator
selections). It uses the five largest observations, averages four excesses, and
is considerably less conservative than the original rule. These tuned choices
do not inherit the paper's confidence or approximation guarantee.

### 2. Recent record improvements

Use the increase in the running best over the last w observations, divided by w,
times a multiplier. Both incumbents are evaluated against the same current
reference. Training chooses w in {4,8,16} and multiplier in {0.5,1,2}, after at
least 2w samples. A window without improvement produces zero estimated gain and
stops. This family performs poorly; lack of recent progress does not reliably
mean another response has little value.

### 3. Remaining utility range

Use multiplier * (1 - current_best_utility)/(n+1), with multiplier in
{0.03,0.1,0.3,1} and minimum count in {4,8}. This is a heuristic allowance for
unobserved improvement, **not a valid confidence bound**.

Its strong result needs interpretation. Before n=100, the prefix empirical
0.99 quantile equals the current maximum, so its estimated best BT utility is
exactly 0.5. Its stopping condition is then equivalent to

    cumulative monetary cost >= multiplier * n / (2*(n+1)).

It therefore behaves almost like a fixed spending budget in that range, rather
than using quality information to decide when to stop. The combined selector
chooses this family in 144/150 settings. The length-budget control was added to
expose this distinction, not to hide it.

### Cost-only control

Training chooses a cumulative response-length budget from 64 geometrically
spaced values between 256 and 2,000,000 characters, plus an unlimited budget
(subject to the shared cap). The policy stops after the first response that
reaches the budget. It never uses reward to decide when to stop. It does use
reward to select the final response, just like fixed N. This is a stronger
comparison than fixed N when lengths vary across prompts.

## Evaluation and limitations

- Five generators, the same 100 Alpaca prompts per generator, and 960 cached
  responses per prompt. Replays use a common cap of 512 and eight random orders.
- Five 50/50 prompt splits with seeds 41–45; replay seed 20260922. All policies
  share exactly the same response orders. Fixed N considers every count 1–512.
- The complete 35-policy grid is specified in code and metadata. Selection uses
  training prompts only, separately for each price and generator. No coefficient
  or family was retuned after inspecting this study's test outcomes.
- Prompt-level replay averages are the units for paired bootstrap intervals.
  Splits overlap and generators share prompts: 150 setting–split rows are not
  150 independent experiments. Reported positive cells average across splits.
- These prompts were already examined in earlier experiments. New split seeds
  do not create independent confirmation; results remain exploratory. No claim
  of superiority over the old fitted-tail policy is made without a paired run.
- Prices charge actual recorded characters, not newly measured model tokens.
- DMRL is unverified, utility estimates change with the reference, costs are
  estimated, and sampling is without replacement. No theorem is transferred.
- At the cheapest price, the cap limits both methods. Near-equality there is not
  evidence of equivalence for an unlimited generation budget.
- The manuscript and all earlier experimental artifacts remain unchanged.

## Reproduction and artifacts

From the adaptive-BoN repository root, with NumPy installed:

```sh
python codex_results/distribution_free_dmrl/nonparametric_study.py \
  --output /path/to/new-study-directory
python codex_results/distribution_free_dmrl/nonparametric_controls.py \
  /path/to/new-study-directory
python -m unittest discover -s codex_results/distribution_free_dmrl -p 'test_*.py'
```

Both runners refuse to overwrite existing results. The study produces
`comparisons.csv`, `prompt_metrics.csv`, complete prompt-level replay arrays,
and `METHOD.json` with grid, seeds, input hashes, and source hash. The control
has its own `cost_only_control/` directory. Costs, utility, profit, and sample
counts are recorded separately. The prior experiment remains in `../results/`.
