# Isotonic probability-space target quality

| Target | Adaptive accuracy | Adaptive chars | Train-tuned N / test accuracy | Saving vs train N | Held-out matched N / accuracy | Saving vs matched N |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.2602 | 2967 | 13.0 / 0.2655 | +54.04% | 8.9 / 0.2636 | +34.18% |
| 0.30 | 0.2941 | 6964 | 58.1 / 0.3099 | +75.70% | 19.9 / 0.2957 | +30.12% |
| 0.35 | 0.3481 | 41620 | 170.6 / 0.3527 | +50.87% | 132.3 / 0.3477 | +36.76% |

## Integrity

Every adaptive result deploys one configuration and one cost divisor. There is no policy mixture, target margin, Fixed-N guard, Fixed-N cap, or Fixed-N fallback. Isotonic reward calibration uses only the 41-problem training half; the corresponding 42-problem test half is untouched until final evaluation.

The first comparator chooses the minimum-character integer N reaching the requested target on train and freezes it for test. The second is a descriptive held-out oracle: the minimum-character integer N whose exact held-out accuracy is at least the adaptive policy's achieved accuracy. Neither comparator affects adaptive stopping.

Splits: 95--104; train/test permutations: 16/48.

The policy selector was frozen from development split IDs 55--59. It places 10% weight on current train cross-fit accuracy and the remaining weight on development held-out accuracy, ranks feasible rules by development geometric-mean characters, and compares the result to the requested target with zero offset. Split IDs 95--104 were not used to choose this selector.
