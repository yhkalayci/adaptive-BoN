# Winner-calibrated coding: completed local experiment

Ten splits, 37 prices, top-three/full-history stopping, plain mean cost,
minimum counts 10–512 and cap 512. The map is one fixed function per split,
selected by four-fold problem-grouped cross-validation inside calibration.
No final-test outcomes choose maps or policy parameters. This is nevertheless
exploratory evaluation on a previously reused cohort, not fresh validation.

## Calibration result

Held-out winner-mixture Brier loss improves from **0.23308 to 0.21695**.
Mean predicted versus actual winner correctness on test problems:

| Count | Actual | Original prediction | New prediction |
|---:|---:|---:|---:|
| 10 | 46.66% | 55.50% | 46.91% |
| 32 | 51.54% | 65.02% | 55.77% |
| 128 | 55.25% | 72.18% | 62.74% |
| 512 | 60.29% | 77.07% | 67.62% |

Calibration improves, but residual high-count overconfidence remains.

## Profit and expenditure

For the new map with predicted-profit parameter tuning:

| Price ($/M tokens) | Profit change vs Fixed-N | Token-cost reduction | Mean calls |
|---:|---:|---:|---:|
| 1.00 | +1.89% | +26.52% | 231.93 |
| 1.25 | +2.65% | +27.72% | 189.20 |
| 2.00 | +0.41% | -19.57% | 130.55 |
| 5.00 | -0.80% | -41.45% | 65.91 |
| 8.00 | -1.12% | -51.77% | 46.81 |
| 10.00 | -1.39% | -53.14% | 39.60 |

Negative cost reduction means spending more at the same price; these are not
matched-quality savings. Better calibration alone does not solve the
high-price profit problem. Correctness-tuned and frozen-setting arms, including
negative outcomes, are preserved in their own folders.

## Contents and reproduction

- `profiles/`: ten fitted maps, compatible with `CodingProfile.load`.
- `map_choices.json`, `map_cv_metrics.csv`: calibration-only map selection.
- `map_quality_metrics.csv`: exact winner calibration, by problem and count.
- `selected_policies.json`: all three arms' policy settings.
- Each arm: `summary.csv`, `split_metrics.csv`, `problem_metrics.csv`.
- Summaries include conditional pointwise 95% paired problem-bootstrap profit
  intervals; these are not selection-adjusted or simultaneous.
- `validation.json`: original map and Fixed-N reproduction, 1,110 public-policy
  replay checks passed.

See `../coding_scripts/SERVER_WINNER_CALIBRATION.md` for the server command.
The entrypoint's `--data` override was added for portability after this run
started; the recorded execution hash predates that CLI-only change. Numerical
fitting and evaluation logic are unchanged. The manuscript was not edited.
