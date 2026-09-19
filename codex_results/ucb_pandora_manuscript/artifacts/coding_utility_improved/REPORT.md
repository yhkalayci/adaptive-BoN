# Improved coding utility-gap result

## Outcome

The profiled utility-frontier method has positive mean held-out utility gain
at every reported character divisor.  It improves the mean relative gain over
the retained shifted-exponential result at all ten divisors and removes all
four negative mean points from that result.

| Utility divisor | New relative gain | Retained relative gain | New additive gap (95% CI) | Positive splits |
|---:|---:|---:|---:|---:|
| 100,000 | 9.49% | 5.75% | 0.01927 [0.01209, 0.02644] | 10/10 |
| 200,000 | 4.90% | 2.00% | 0.01132 [0.00773, 0.01491] | 10/10 |
| 300,000 | 2.50% | 1.85% | 0.00598 [0.00034, 0.01161] | 9/10 |
| 400,000 | 1.37% | 0.69% | 0.00289 [-0.00275, 0.00852] | 8/10 |
| 500,000 | 3.42% | -0.11% | 0.00843 [0.00149, 0.01537] | 9/10 |
| 600,000 | 2.84% | 0.59% | 0.00662 [-0.00153, 0.01476] | 8/10 |
| 700,000 | 1.34% | -0.13% | 0.00240 [-0.00773, 0.01253] | 7/10 |
| 800,000 | 1.14% | 0.42% | 0.00188 [-0.00778, 0.01155] | 7/10 |
| 900,000 | 1.15% | -0.76% | 0.00195 [-0.00823, 0.01213] | 6/10 |
| 1,000,000 | 0.93% | -1.24% | 0.00143 [-0.00906, 0.01192] | 6/10 |

Across the 100 split-by-divisor cells, 80 have positive additive gain, versus
62 for the retained method.  The simple mean of split-level relative gains
over all ten divisors is 2.91%, versus 0.91% for the retained method.

The uncertainty still matters: the additive-gain interval is strictly above
zero at 100k, 200k, 300k, and 500k.  The remaining means are positive, but
their split-level intervals include zero.

## Algorithm retained

Every final policy is an uncapped shifted-exponential UCB-Pandora rule on
outer-train-only isotonic correctness probabilities.  Every confidence scale
is strictly positive.  The reward tail is truncated to `[0, 1]`; the next
character count is estimated without seeing the next response; realized cost
is the exact cumulative number of opened characters.

The improvement comes from calibrating the reservation comparison separately
from the reporting price.  For actual utility divisor `D`, reservation divisor
`D_res`, prefix length `n`, and minimum opening count 3, the stopping rule is

```text
EI_UCB(n) * (n / 3)^(-tail_decay) > estimated_next_chars / D_res.
```

This is the shared blueprint with

```text
A_n = (D_res / D) * (n / 3)^(-tail_decay),  B_n = 0,
```

so it equivalently compares `A_n * EI_UCB(n)` with
`estimated_next_chars / D`.

## Leakage control

- Split IDs 55--59 supplied the already-frozen target-quality frontier.
- Split IDs 65--69 shortlisted and validated policies using arithmetic
  utility.  A 50/50 profile/validation score and a 0.1 validation-standard-
  deviation penalty were chosen by leave-one-validation-split-out checks.
- One configuration and reservation divisor per utility divisor were then
  frozen in `../frontier_validation/frozen_selection.csv`.
- Split IDs 75--84, with 48 permutations per held-out problem, were used only
  for the final table above.

The Fixed-N comparator is selected from each outer-training half and evaluated
on the same final problems and reward permutations as the UCB-Pandora policy.
