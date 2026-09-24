# Gain and cost optimism audit on token-count alignment data

This paired audit uses four generators, FSFairX and RM-Mistral, 100 prompts per
generator, eight response orders, and five 50/50 prompt splits. The table below
equally averages the 40 generator/reward/price settings at $0.1, $0.2, $1, $2,
and $10 per million output tokens. Each setting is compared with its own single
fixed N selected on training prompts and frozen on test prompts. Percentages
are means of splitwise relative net-utility gains.

| Policy | Cost adjustment beta | Gain factor | Mean samples | Mean profit gain vs fixed N | Positive settings | Mean estimated-cost discount at stop |
|---|---:|---:|---:|---:|---:|---:|
| Current smoothed upper-half gain | 0 | 1 | 194.1 | +2.78% | 40/40 | 0% |
| Current smoothed upper-half gain | 1 | 1 | 195.9 | +2.90% | 40/40 | 2.6% |
| Current smoothed upper-half gain | 2 | 1 | 197.7 | +3.00% | 40/40 | 4.9% |
| Smoothed top-two theoretical gain | 0 | 4 | 482.3 | -24.10% | 0/40 | 0% |
| Smoothed top-two theoretical gain | 2 | 4 | 485.6 | -25.44% | 0/40 | 2.6% |
| Smoothed top-two diagnostic | 2 | 1 | 259.0 | -0.79% | 10/40 | 4.1% |
| Smoothed top-two diagnostic | 2 | 0.25 | 105.8 | -3.34% | 8/40 | 6.3% |

The **theoretical factor 4 applied to the top-two gap is the dominant source of
excessive continuation** in this plug-in. Removing cost optimism from that rule
changes its mean profit gain only from -25.44% to -24.10%; it remains below
fixed N in all 40 settings. In the current empirical rule, beta=2 changes the
mean from +2.78% to +3.00%, a small point-estimate difference that is not a
confidence claim. The alternate factors 1 and 0.25 are diagnostics, not
theorem-equivalent replacements; neither matches the current policy on average.

For Qwen 3.5 9B with RM-Mistral at $10/M tokens:

| Policy | Mean samples | Profit gain vs fixed N | Mean cost discount at stop |
|---|---:|---:|---:|
| Current, beta=0 | 13.3 | +5.60% | 0% |
| Current, beta=2 | 14.6 | +5.46% | 9.0% |
| Top-two factor 4, beta=0 | 66.5 | -48.75% | 0% |
| Top-two factor 4, beta=2 | 69.9 | -52.27% | 4.8% |

All variants use the same token records. Within each policy family, the utility
transform and gain statistic stay fixed while beta or the gain factor changes.
The current family uses an upper-half tail benchmark; the top-two family uses
sigmoid reward relative to the observed-prefix q99. Thus comparisons **between**
families also reflect different transforms and tail statistics. The script
asserts that its current-beta2 and top-two-factor4-beta2
arrays exactly reproduce the corresponding saved replay arrays. The moving
prefix utility reference, variable token costs, and finite cap mean the
top-two plug-in is not the theorem's exact setting. The q99 benchmark
extrapolation in the current gain formula was not separately ablated here, so
this audit does not establish whether that component is optimistic.
