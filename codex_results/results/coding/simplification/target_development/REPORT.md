# Isotonic probability-space target quality

| Target | Adaptive accuracy | Adaptive chars | Train-tuned N / test accuracy | Saving vs train N | Held-out matched N / accuracy | Saving vs matched N |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.2851 | 5977 | 18.6 / 0.2815 | +35.68% | 17.8 / 0.2877 | +32.34% |
| 0.30 | 0.3119 | 14089 | 71.0 / 0.3181 | +59.95% | 43.0 / 0.3137 | +35.66% |
| 0.35 | 0.3464 | 31516 | 147.4 / 0.3489 | +57.18% | 98.8 / 0.3468 | +38.01% |

## Integrity

Every adaptive result deploys one configuration and one cost divisor. There is no policy mixture, target margin, Fixed-N guard, Fixed-N cap, or Fixed-N fallback. Isotonic reward calibration uses only the 41-problem training half; the corresponding 42-problem test half is untouched until final evaluation.

The first comparator chooses the minimum-character integer N reaching the requested target on train and freezes it for test. The second is a descriptive held-out oracle: the minimum-character integer N whose exact held-out accuracy is at least the adaptive policy's achieved accuracy. Neither comparator affects adaptive stopping.

Splits: 55--59; train/test permutations: 4/8.
