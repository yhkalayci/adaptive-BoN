# Isotonic probability-space target quality

| Target | Adaptive accuracy | Adaptive chars | Train-tuned N / test accuracy | Saving vs train N | Exact-quality fixed expected N / accuracy | Saving at equal quality |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.2602 | 2967 | 13.0 / 0.2655 | +54.04% | 8.2 / 0.2602 | +28.26% |
| 0.30 | 0.2953 | 7334 | 58.1 / 0.3099 | +74.41% | 18.6 / 0.2953 | +22.24% |
| 0.35 | 0.3462 | 38823 | 170.6 / 0.3527 | +54.17% | 77.5 / 0.3462 | +1.13% |

## Integrity

Every adaptive result deploys one configuration and one cost divisor. There is no policy mixture, target margin, Fixed-N guard, Fixed-N cap, or Fixed-N fallback. Isotonic reward calibration uses only the 41-problem training half; the corresponding 42-problem test half is untouched until final evaluation.

The first comparator chooses the minimum-character integer N reaching the requested target on train and freezes it for test. The second is a descriptive held-out oracle at exactly the adaptive policy's aggregate accuracy. It uses one common integer N when possible and otherwise randomizes between two common fixed counts before generation, choosing the minimum-cost exact-quality mixture on the averaged held-out curve. Neither comparator affects adaptive stopping.

Splits: 95--104; train/test permutations: 16/48.

The policy selector was frozen from development split IDs 55--59. It places 0% weight on current train cross-fit accuracy and the remaining weight on development held-out accuracy, ranks feasible rules by development geometric-mean characters, and compares the result to the requested target with zero offset. Split IDs 95--104 were not used to choose this selector.
