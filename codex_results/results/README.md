# Results

The active results tree contains alignment experiments, the strongest
confirmed coding experiments, and one cross-task response-length check.

## Target quality

`coding/target_quality` is the successful uncapped,
single-policy target-quality result. It uses train-only isotonic reward
calibration, bounded Gaussian or shifted-exponential fits, and a smooth
expected-improvement tail decay. There is no Fixed-N cap, guard, fallback,
mixture, or target margin.

| Target | Accuracy | Characters | Saving vs matched Fixed-N |
|---:|---:|---:|---:|
| 0.25 | 0.2503 | 2,371 | +33.86% |
| 0.30 | 0.2911 | 6,576 | +29.41% |
| 0.35 | 0.3417 | 36,493 | +37.62% |

## Utility gap

`coding/utility_gap` is the successful calibrated utility
confirmation. Its clearest relative utility gains over train-tuned Fixed-N
are `+5.93%`, `+1.79%`, `+1.91%`, and `+0.71%` at character divisors
100k--400k. It also reports accuracy at exactly matched expected character
cost.

Superseded base runs, rejected development variants, partial files, and
notebook caches are intentionally excluded from `codex_results/results`.

## Response-length distribution check

`distribution_check` contains a 1x3 histogram comparing per-problem average
response characters for Alpaca, HH-RLHF, and Coding. Alpaca and HH-RLHF use
Llama-3.1-8B only. The corresponding problem cohorts contain 100, 100, and 83
problems, with 960, 960, and 512 responses per problem respectively.
