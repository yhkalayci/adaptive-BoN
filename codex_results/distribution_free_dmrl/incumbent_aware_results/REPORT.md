# Paired incumbent-aware stopping experiment

## Coding

No coding responses were available. No coding claim or new generation is made.

## Alignment: a small additional gain, not a substantial breakthrough

This experiment tests whether a strong incumbent should reduce continuation
more explicitly than in the earlier order-statistic spacing rule. All methods
share the previous study's exact prompts, splits, permutations, prices, and
960-response cap. The prior spacing arrays are reproduced and checked to
floating-point tolerance for every generator and split.

| Training-selected family | Profit gain vs training-selected fixed N | Matched-quality saving |
|---|---:|---:|
| Previous spacing rule | 4.12% | 20.67% |
| DMRL exponential-envelope plug-in | −0.22% | −6.90% |
| Second-moment-envelope plug-in | 3.25% | 16.13% |
| Training-selected choice among all three | **4.16%** | **20.81%** |

Adding both families improves the overall point estimates by only 0.033
percentage points of relative profit and 0.138 percentage points of cost saving.
This is not evidence of a meaningful improvement and does not justify claiming
a breakthrough. The previous simpler method remains a reasonable default.
The combined selector chooses spacing in 112/150 profit settings, the moment
envelope in 32, and the DMRL envelope in six.

Profit gains for the combined selector are positive in 27/30 generator–price
means. Matched-quality savings are positive in 34/35 generator–target means,
versus 35/35 for the previous rule. One target is therefore worse despite the
slightly larger overall average. The paired cost-only control from the previous
study still does better: 6.38% profit gain and 30.02% matched-quality savings.

The second-moment family improves some weaker cases: Mistral's averages change
from 1.32% profit gain / 7.53% saving with spacing to 1.85% / 10.09%. But selecting
that family after inspecting those test outcomes would not be legitimate;
the training-selected combined method achieves 1.29% / 9.04% for Mistral.

## Algorithms

The previous spacing rule estimates improvement from the average excess of the
largest k observations over the next order statistic, divided by sample count.
An unusually high best observation can increase that estimate. The two new
variants instead estimate tail scale from the next k observations, excluding
the best. Let xi be the (k+2)nd largest utility, M the current maximum, and d=M-xi.
Let m and q be empirical first and second moments of those k excesses above xi.
Use (k+1)/n as the empirical fraction above the threshold.

1. **DMRL envelope:** estimated gain is (k+1)/n * m * exp(-d/m). For a genuinely
   DMRL conditional excess with known mean, the exponential envelope follows
   from the shape restriction; it does not require an exponential law. However,
   the empirical estimates here are not valid confidence bounds. The expression
   is algebraically similar to an exponential-tail plug-in, which we do not
   disguise as an unrelated technique. Zero empirical scale gives zero gain.
2. **Second-moment envelope:** estimated gain is
   (k+1)/n * min(m, q/(4d)). The inequality (X-d)+ <= X^2/(4d) motivates the
   second term for d>0. Tied observations give zero gain. No response law is
   fitted. Replacing population moments with order-statistic estimates again
   makes this a heuristic, not a certified bound.

Both stop when the gain estimate falls below stopping-price times the observed
mean response length. The current best is still retained for output and all
generated responses are charged. Excluding the incumbent from scale estimation
is intended to avoid treating an unusually good answer as a reason to continue.
The DMRL-envelope results show that this intuition alone is insufficient: the
plug-in can be too eager to stop. One of its 175 training target settings cannot
reach the requested quality; that failure is retained in the results.

The five width rules and 49 stopping prices are unchanged. Both empirical and
cross-fitted bias-corrected BT references are considered. Family, reference,
width, and stopping price are selected using training prompts only. Target
mixtures are selected on training data, then held fixed; the test comparator
matches attained quality using the cheapest fixed-count mixture.

## Cost-only policy clarification

The control selects one cumulative response-length budget for each generator
and price using training data. It then applies that budget to every test prompt.
It does **not** assume that individual responses have equal cost. Short responses
permit more samples and long responses fewer. Stopping is after a completed
response, so final cost may overshoot the budget. Rewards select the output but
do not affect stopping. This contrasts with the order-statistic policies, which
share trained parameters but estimate next cost separately from each prompt's
observed mean response length and do not impose one fixed spending budget.

## Scope, checks, and reproduction

- Five generators, 100 shared prompts, five 50/50 splits, eight paired response
  orders, six actual prices, seven quality targets, and a 960-response cap.
- Prespecified extra families; no coefficient was retuned after test inspection.
- Reused prompts make this exploratory development, not independent confirmation.
- Costs use the available character lengths, not new token measurements.
- The paper's theorem is not asserted for any plug-in or tuned variant.
- Previous results, source policies, and manuscript remain unchanged.
- Four additional tests verify incumbent responsiveness, equivalence to previous
  spacing replay, prefix-only features, and isolation of evaluation quantities.
- Full records are in `profit.csv`, `target_quality.csv`, and the replay arrays.
  `AUDIT.json` records independent selection and linear-programming checks.

From the repository root:

```sh
python codex_results/distribution_free_dmrl/incumbent_aware_study.py \
  --output /path/to/new-incumbent-study
python codex_results/distribution_free_dmrl/audit_refinement.py \
  /path/to/new-incumbent-study
```
