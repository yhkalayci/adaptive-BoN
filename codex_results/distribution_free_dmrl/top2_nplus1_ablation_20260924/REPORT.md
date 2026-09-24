# Smoothed top-two gain: n versus n+1 denominator

The replay uses the same token-count records, five 50/50 prompt splits, eight
response orders, and train-selected fixed-N comparator as the current alignment
report. The gain statistic is the top-two utility excess above the third-largest
with the same observed-prefix q99 sigmoid transform. The cost estimate remains
`price*mean_output_tokens/(1+2*SE/mean_output_tokens)` for every variant.

Two readings of “divide by n+1” are tested:

- **Literal denominator change:** `4*mean(m_4,...,m_n)/(n+1)` instead of
  `4*mean(m_4,...,m_n)/n`.
- **Replace factor 4/n:** `mean(m_4,...,m_n)/(n+1)`.

For completeness, `record_then_smooth` uses the current policy's smoothing
pattern on top-two excesses: `mean(j*m_j/(j+1), j=4,...,n)/n`. This is distinct
from either direct denominator change.

The table equally averages the 40 generator/reward/price settings at $0.1,
$0.2, $1, $2, and $10 per million output tokens. Gains are means of splitwise
relative net-utility gains over each train-selected fixed N.

| Policy | Mean samples | Mean profit gain | Positive settings | Settings beating current |
|---|---:|---:|---:|---:|
| Top-two `4/n` | 485.57 | -25.44% | 0/40 | 0/40 |
| Top-two `4/(n+1)` | 484.97 | -24.91% | 0/40 | 0/40 |
| Top-two `1/n` | 259.03 | -0.79% | 10/40 | 0/40 |
| Top-two `1/(n+1)` | 258.24 | -0.66% | 12/40 | 0/40 |
| Top-two record-then-smooth | 255.55 | -0.54% | 12/40 | 0/40 |
| Current `mean_costse2` | 197.65 | +3.00% | 40/40 | — |

For Qwen 3.5 9B with RM-Mistral:

| $/M tokens | Top-two `4/(n+1)` calls | Gain | Top-two `1/(n+1)` calls | Gain | Current calls | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 942.05 | -2.00% | 652.84 | +0.17% | 536.37 | +1.39% |
| 0.2 | 837.85 | -7.55% | 463.74 | -0.53% | 360.27 | +1.70% |
| 1 | 409.50 | -22.18% | 144.91 | -1.29% | 106.60 | +2.70% |
| 2 | 254.76 | -29.25% | 82.39 | -1.40% | 58.72 | +2.00% |
| 10 | 68.78 | -50.26% | 22.69 | +0.36% | 14.59 | +5.46% |

The literal n+1 change is small because n is usually large when the policy
stops. Removing the factor 4 makes a much larger difference, but the resulting
top-two rule still samples more and yields lower profit than the current rule
in all 40 compared settings. The current rule also uses a different tail width
and Bradley–Terry gain estimate, so this result does not isolate a single
mechanism behind its advantage.

The replay asserts that its `4/n` and `1/n` arrays match the prior saved token
ablations. See `summary.csv` for every generator/reward/price and
`split_results.csv` for all held-out split values. The observed-prefix utility
reference, random token lengths, and finite cap keep these policies outside
the theorem's exact assumptions.
