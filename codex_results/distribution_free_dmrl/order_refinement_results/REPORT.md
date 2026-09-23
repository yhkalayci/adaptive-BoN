# Refined order-statistic stopping: profit and target quality

## Coding

The coding response cache is still absent, so this report contains no new coding
performance claim. No generation was run. The existing coding adapter remains
available; the present refinement study evaluates alignment only.

## Alignment results

The refined order-statistic method beats training-selected fixed N by **4.12%**
in mean relative profit. It saves **20.67%** of generation cost against the
cheapest fixed-N mixtures retrospectively matched to its attained quality.
No parametric response or utility distribution is fitted.

An independent audit in `AUDIT.json` checked 450 training-only profit
selections, 525 training target-mixture optima, and 525 test quality-matched
optima. Mixture costs were verified with a separate linear-program solver.
Shared-prompt bootstrap sensitivity intervals, conditional on the fitted and
selected policies, are 2.85–5.55% for profit improvement and 13.91–25.07% for
matched-quality savings. They account for shared prompt weights across splits
and generators, but are not independent confirmation or adjustment for
repeated model development.

| Variant | Profit improvement vs fixed N | Matched-quality cost saving |
|---|---:|---:|
| Order statistics with empirical prefix reference | +4.01% | 19.78% |
| Order statistics with training-corrected reference | +4.08% | 20.56% |
| Training-selected choice of these two variants | **+4.12%** | **20.67%** |
| Paired cost-only length-budget control | +6.38% | 30.02% |

The combined method is selected using training data, not by choosing the
test winner. Its mean relative profit improvement is positive in 27/30
generator–price settings (after averaging the five splits). Matched-quality
savings are positive in all 35 generator–target settings. These are descriptive
point estimates, not 35 independent significance tests.

**Important qualification:** the method improves on fixed N, but still does not
beat the cost-only length-budget control. This study does not establish that
its reward-dependent decisions are more useful than adapting expenditure to
response length. It also does not establish superiority to the manuscript's
old fitted-tail implementations, which were not rerun in this paired study.

### Results by generator

Training-selected variant, averaging across prices or targets and splits:

| Generator | Profit improvement | Matched-quality cost saving |
|---|---:|---:|
| Gemma-2-9B | +3.21% | 17.18% |
| Llama-3.1-8B | +6.38% | 28.52% |
| Llama-3.2-3B | +5.40% | 26.82% |
| Mistral-7B | +1.32% | 7.53% |
| Qwen-2.5-7B | +4.30% | 23.29% |

### Target quality and cost saving

The requested targets are proxy BT utilities, not validated human-preference
probabilities. Reported savings match **attained test quality**, not the nominal
target. The adaptive policies themselves are selected on training prompts.

| Requested quality | Mean attained quality | Matched-quality cost saving |
|---:|---:|---:|
| 0.30 | 0.308 | 12.53% |
| 0.35 | 0.351 | 19.17% |
| 0.40 | 0.401 | 25.09% |
| 0.45 | 0.450 | 24.69% |
| 0.50 | 0.497 | 20.97% |
| 0.55 | 0.547 | 21.50% |
| 0.60 | 0.596 | 20.71% |

These averages do not guarantee target attainment in every setting. For
example, at target 0.60, individual generator–split attained qualities range
from 0.568 to 0.627; at target 0.45 the range is 0.419 to 0.480. All requested
targets were attainable on training data, and all matched-quality comparisons
were feasible on test data; no extrapolation was used.

## The algorithm and what it needs

An order statistic is simply an observation's position in a sorted sample.
At count n, sort the current utility estimates as v1 >= v2 >= ... and calculate

    mean_excess = mean(v1 - v[k+1], ..., vk - v[k+1])
    estimated_improvement = mean_excess / n

The policy stops when estimated improvement is at most

    stopping_price * observed_mean_response_length.

Equivalently, a trained multiplier can scale estimated improvement before
comparing it with the actual estimated generation cost. The price used to
control stopping is not necessarily the actual price charged in profit.
Every opened response is charged at the actual price. All policies check after
every response and return the largest raw reward among opened responses.

We broaden the original top-three statistic using five choices:

- fixed k=2, 4, or 8, with minimum count max(4,k+1);
- k=min(32,ceil(sqrt(n))), at least 2, after four responses;
- k=min(32,ceil(n/4)), at least 2, after four responses.

Larger k uses more tail observations and reduces reliance on a single small
gap. Among 150 profit selections, the combined selector chooses the quarter
rule 59 times, fixed k=8 58 times, square root 28 times, fixed k=4 four times,
and fixed k=2 once. The new policy remains explicitly reward-dependent and
contains no additive fixed-budget term or remaining-range policy.

### Utility reference

In the basic variant, rewards are transformed as sigmoid(reward - prefix_q99).
The reference uses only responses observed so far. No training calibration is
needed for this transform. Training still selects k's rule and stopping price.

The optional corrected variant adds an empirical bias table b[n] to prefix_q99.
For each n, b[n] is the average difference between a training prompt's full-pool
0.99 quantile and its n-sample quantile. This is a direct quantile-bias estimate,
not a fitted distribution. Training evaluations cross-fit the table: each half
of the outer training prompts is evaluated with a table from the other half.
Test evaluation uses a table from the whole outer training half. No test-prompt
full-pool reference enters any stopping decision.

The correction contributes only a modest gain relative to the simpler variant
(4.01% versus 4.12% profit improvement; 19.78% versus 20.67% matched saving).
The simple variant may therefore be preferable when implementation simplicity
is the priority. The experiment does not prove why either component helps.

### Deployment inputs

1. A prompt, generator, and reward scorer.
2. A defined score-to-utility transformation (here the BT proxy).
3. Observed response lengths, plus actual price for accounting.
4. A frozen width rule, stopping price, minimum count, and cap.
5. Only for the corrected version, the training-derived reference-bias table.

No hidden labels, future samples, full test-prompt quantile, distributional fit,
or known mean cost is required online. Training caches are needed offline to
select the practical parameters and to calibrate the optional bias table.
For target quality, training selects a mixture of at most two stopping policies;
deployment draws which policy to run before observing responses.

## Evaluation protocol

- Five alignment generators, each with the same 100 Alpaca prompts and 960
  cached responses scored by mistral_rm_reward.
- Full 960-response cap, eight response permutations per prompt, and replay
  seed 20260923. All methods use identical response orderings within a prompt.
- Five 50/50 prompt splits with seeds 71–75. The 49 stopping-price values,
  logarithmically spaced from 2e-10 to 2e-4, the five width rules, and both
  reference variants are specified in metadata before evaluation.
- Profit selection maximizes training profit over these configurations. The
  fixed-N baseline maximizes training profit over **every N from 1 to 960**.
- Target-quality selection uses the lowest-cost convex mixture of adaptive
  policies attaining the target on training data. Its mixture remains frozen
  on test data; no test quality is used to select the adaptive policy.
- The retrospective comparator is the cheapest mixture over every fixed N
  matching the adaptive policy's mean attained test quality. This comparator
  is an analytical benchmark, not a deployable test-free baseline. Its lower
  convex envelope is verified against linear-programming solutions in tests.
- `target_quality.csv` also records the training-selected fixed-N mixture and
  its attained quality. Savings against that comparator are not described as
  quality matched when its attained quality differs.
- The cost-only control chooses a cumulative length budget on training data
  from a prespecified geometric grid. It never uses reward for stopping.
- Existing text character counts are the cost units. These are not new
  tokenizer-based results. The token-count rerun remains future work.

## Interpretation and limitations

The broader order-statistic family is substantially more competitive than the
original largest-three rule, while keeping the same estimated-improvement
versus-cost structure. It reaches roughly 20% matched-quality savings overall
and around 25% at targets 0.40–0.45. It is not uniformly better: three price–
generator means do not improve profit, and Mistral savings are much smaller.

The preceding study's 2.50% order-statistic result used a 512 cap and different
splits/orders. It is not a paired ablation of this 4.12% result. The empirical-
versus-corrected reference comparison in this report is paired.

These are exploratory refinements on previously examined prompts. New seeds
do not make the prompts independent confirmation data. Splits overlap and
generators share prompts; there are not 500 independent questions or 150
independent profit experiments. Profit CSVs contain paired prompt-bootstrap
intervals conditional on a training split; these are pointwise and do not
correct for repeated model development. No overall significance claim is made.

The theorem is not transferred to these implementations: k may change with n,
the scale is tuned, checks are sequential, utility estimates change with the
reference, costs are estimated, and DMRL is unverified. Sampling from finite
caches is also without replacement. The manuscript and previous results were
not modified.

## Reproduction

From the repository root, with Python and NumPy:

```sh
python codex_results/distribution_free_dmrl/order_statistic_refinement.py \
  --output /path/to/new-refinement-directory
python codex_results/distribution_free_dmrl/order_refinement_controls.py \
  /path/to/new-refinement-directory
python -m unittest discover -s codex_results/distribution_free_dmrl -p 'test_*.py'
```

SciPy is needed only for the linear-programming unit tests. Existing result
directories are never overwritten. Main outputs are `profit.csv`,
`target_quality.csv`, `target_prompt_metrics.csv`, per-generator/per-split
replay arrays, `METHOD.json`, and the paired `cost_control/` directory.
