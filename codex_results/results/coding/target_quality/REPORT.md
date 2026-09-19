# Isotonic probability-space target quality

| Target | Adaptive accuracy | Adaptive chars | Train-tuned N / test accuracy | Saving vs train N | Held-out matched N / accuracy | Saving vs matched N |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.2503 | 2371 | 13.0 / 0.2655 | +63.28% | 7.1 / 0.2526 | +33.86% |
| 0.30 | 0.2911 | 6576 | 58.1 / 0.3099 | +77.05% | 18.6 / 0.2926 | +29.41% |
| 0.35 | 0.3417 | 36493 | 170.6 / 0.3527 | +56.92% | 117.5 / 0.3415 | +37.62% |

## Integrity

Every adaptive result deploys one configuration and one cost divisor. There is no policy mixture, target margin, Fixed-N guard, Fixed-N cap, or Fixed-N fallback. Isotonic reward calibration uses only the 41-problem training half; the corresponding 42-problem test half is untouched until final evaluation.

The first comparator chooses the minimum-character integer N reaching the requested target on train and freezes it for test. The second is a descriptive held-out oracle: the minimum-character integer N whose exact held-out accuracy is at least the adaptive policy's achieved accuracy. Neither comparator affects adaptive stopping.

Splits: 95--104; train/test permutations: 16/48.

The policy selector was frozen from development split IDs 55--59. It blends
10% current train cross-fit accuracy with 90% development held-out accuracy,
ranks feasible rules by development geometric-mean characters, and compares
the result to the requested target with zero offset. Split IDs 95--104 were
not used to choose this selector.
