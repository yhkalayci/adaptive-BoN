# Alignment UCB-Pandora implementation

## Finding

The current all-generator alignment experiment is a UCB plug-in Pandora
reservation policy. It uses:

1. a shifted-exponential upper-tail fit on exponentiated rewards,
   \(Y=\exp(R)\);
2. Bradley--Terry (BT) utility on raw reward differences,
   \(u(r;b)=\sigma(r-b)\); and
3. an output-character opening cost,
   \(E[\text{next characters}]/D\), where \(D\) is the configured character
   divisor.

The frozen configuration in
`code/alignment_scripts/alignment_pandora_all_generators.py` selects
`exp_tail_exponentiated` with:

- confidence scale: `0.4`
- reward-prior strength: `5.0`
- cost-prior strength: `0.0`
- tail threshold quantile: `0.5`
- benchmark calibration: `0.0`
- EI bonus scale: `0.002`
- minimum initially opened generations: `3`
- confidence parameter `delta`: `0.05` (the default)

## Simplified accurate pseudocode

```text
Fit training prior:
    for each training prompt:
        y = exp(clipped raw rewards)
        L = median(y)
        S = mean(y values at or above L) - L
        record S / L

    prior_scale_ratio = median(recorded ratios)


Run UCB-Pandora on one prompt:
    open the first 3 generations

    while unopened generations remain:
        n = number opened
        best = largest observed raw reward

        # Fit a shifted-exponential tail to exponentiated rewards.
        y_i = exp(clipped observed raw reward_i)
        L = empirical median(y_i)
        tail = values y_i at or above L
        S_data = mean(tail) - L

        # Shrink the scale toward the training prior.
        S_prior = L * prior_scale_ratio
        S = (number_in_tail * S_data + 5 * S_prior)
            / (number_in_tail + 5)

        radius = sqrt(log(1 / delta) / number_in_tail)
        S_UCB = S * (1 + 0.4 * radius)
        S_LCB = max(S * (1 - 0.4 * radius), epsilon)

        # Estimate the raw-reward 99th-percentile benchmark.
        k = -log((1 - 0.99) / 0.5)
        benchmark = log(L + S_LCB * k)

        # Optimistic future tail: E has an Exp(1) distribution.
        future_raw_reward = log(L + S_UCB * E)

        # Bradley--Terry expected improvement.
        current_utility = sigmoid(best - benchmark)
        tail_EI = 0.5 * expectation[
            max(sigmoid(future_raw_reward - benchmark)
                - current_utility, 0)
        ]

        EI_UCB = min(
            tail_EI
            + 0.002 * sqrt(log(1 / delta) / (2 * n)),
            1
        )

        expected_next_chars = total observed output characters / n
        next_box_cost = expected_next_chars / character_divisor

        if EI_UCB <= next_box_cost:
            stop and return the best observed generation
        else:
            open the next generation

    report the best raw reward and the exact cumulative output characters
```

The expectation above is evaluated deterministically using 48 midpoint
quantiles of the unit exponential distribution. The factor `0.5` is the
probability mass assigned to the fitted upper tail. The fitted body contributes
zero positive improvement because its values cannot exceed the already
observed maximum.

## Character-to-cost behavior

Each generation's realized character count is computed as
`len(g["text"])`. Thus the implementation counts output text using Python
string length; it does not count tokens, bytes, or prompt characters.

There are two distinct cost quantities:

- **Stopping-time estimate:** expected next characters divided by the
  character divisor.
- **Reported realized cost:** the exact sum of output characters for every
  opened generation, divided by the character divisor when utility is
  computed.

Although the code can blend a prompt-feature character prior with observed
lengths, the current frozen alignment configuration sets
`cost_prior_strength=0.0`. Therefore its stopping-time estimate is simply the
mean character length of the generations already opened:

\[
\widehat C_{n+1}=\frac{1}{n}\sum_{i=1}^{n} C_i.
\]

The exact length of an unopened generation is never used before that generation
is opened.

## BT transformation and evaluation

The shifted-exponential fit is performed in exponentiated-reward space. A
future tail value is then mapped back to raw reward with `log`, after which BT
utility is computed as

\[
\sigma(R_{\mathrm{future}}-b)
\quad\text{and}\quad
\sigma(R_{\mathrm{best}}-b).
\]

Thus BT is not applied directly to the exponentiated reward. It is applied to
the difference between raw reward and an estimated raw-reward benchmark.

For held-out scoring, quality is
\(\sigma(R_{\mathrm{best}}-R_{q_{0.99}})\), where \(R_{q_{0.99}}\) is the
prompt's empirical 99th-percentile reward. That full-prompt empirical quantile
is used only after stopping for evaluation and is not passed to the policy.

## Interpretation

The stopping rule has the Pandora reservation/fair-cap form

\[
\operatorname{EI}_{\mathrm{UCB}}(\text{best})
\leq
\frac{E[\text{next characters}]}{D}.
\]

Equivalently, the policy stops when the current BT utility exceeds the
reservation value induced by the optimistic fitted reward law and the estimated
one-box cost.

This should be described as a **UCB plug-in Pandora policy** rather than the
classical known-distribution Pandora model: the common reward distribution and
next-box character cost are re-estimated from the observed prefix as boxes are
opened. All unopened generations are treated as exchangeable boxes.

## Verification

The four focused checks in
`code/alignment_scripts/test_alignment_pandora_ucb.py` pass when executed
directly. They verify, among other properties, that:

- reported characters equal the exact opened-prefix character sum;
- an unseen suffix cannot alter an earlier stop; and
- the shared multi-divisor sweep matches independent Pandora runs.
