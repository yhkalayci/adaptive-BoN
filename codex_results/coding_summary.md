# Coding algorithm summary

## Bottom line

**Follow-up:** the isolated profiled utility-frontier experiment under
`coding_utility_improvement/results/frontier_final` supersedes the retained
utility-gap numbers when the goal is maximum utility.  It requires strictly
positive UCB optimism, remains within the shifted-exponential Pandora
blueprint, and reports positive mean gains at all ten divisors (9.49% at
100,000, 4.90% at 200,000, and 0.93--3.42% thereafter).  The retained results
below remain unchanged for provenance.

The retained coding method is an **unknown-distribution, unknown-next-cost
Pandora-style stopping policy** operating on a train-fitted isotonic estimate
of correctness probability and using a shifted-exponential model for the
unknown upper tail.

It is not accurate to call every retained policy “UCB Pandora”:

- If **confidence_scale > 0**, the fitted tail location and scale are made
  optimistic, so the policy is UCB-style Pandora.
- If **confidence_scale = 0**, the policy is a prior-regularized plug-in
  Pandora rule with no UCB optimism.

In the saved utility experiment, 56 of 100 shifted-exponential selections use
positive optimism and 44 use zero optimism. In the saved target experiment,
27 of 30 use positive optimism and three use zero optimism.

The paper-ready, implementation-matched account is in
[algorithm_section.tex](algorithm_section.tex). This summary deliberately
focuses on the retained shifted-exponential coding family.

## Simplest accurate pseudocode

    Fit monotone P(correct | raw reward) on outer-training problems.
    Transform training rewards to correctness probabilities.
    Fit training priors for upper-tail location, upper-tail scale, and characters.

    For a new problem:
        open 3 candidates

        while the experiment horizon has not been reached:
            update the local upper-tail fit from opened probabilities
            shrink it toward the training prior
            optionally make its location and scale optimistic
            estimate expected improvement over the best opened probability
            optionally decay that expected improvement with the prefix length
            estimate the next output length from opened lengths and the cost prior

            if adjusted expected improvement <= estimated next characters / D:
                stop

            open one more candidate

        return the first opened candidate attaining the largest calibrated
        correctness probability

The returned prefix is charged its exact cumulative output characters.
Vectorized evaluation does not reveal future values to the rule: the decision
at prefix **n** uses only the first **n** rewards and lengths.

## Unknown-distribution, unknown-cost Pandora definition

Each candidate generation is a box. Opening it reveals:

1. a value, represented online by its calibrated probability of correctness;
2. a cost, its output character count.

The next value distribution and next character count are unknown before the
box is opened. The algorithm estimates both from training problems and the
opened prefix. If the fitted value law is **F_n** and the best probability seen
so far is **b_n**, the core reservation comparison is

    adjustment × E[(next_probability - b_n)+]
        > estimated_next_characters / D.

There is no choice among differently distributed boxes: future generations
are treated as exchangeable, so the only adaptive choice is stop versus open
one more. **D** is an externally selected conversion factor, not the unknown
cost; the unknown cost is the next response length.

## Utility and cost calibration

### Utility fit

For each outer split:

1. Split the 83 solvable problems into 41 train and 42 untouched test
   problems.
2. Pool all 41 × 512 = 20,992 **(raw reward, correctness)** training pairs.
3. Fit an equally weighted, nondecreasing isotonic regression
   **g_s(raw reward) = estimated P(correct)**.
4. Clip out-of-range scores to the fitted endpoint probabilities.
5. Freeze **g_s** and apply it to both training and final-test rewards.

The final 42 test problems do not affect this map. One qualification matters:
the map is fit once on all 41 outer-training problems before four-fold policy
selection. The distribution prior is refit inside each fold, but this
isotonic map is not. Thus “train-fitted calibration with held-out final
evaluation” is precise; “fully cross-fitted calibration” is not.

### Distribution and cost-prior fit

For each training problem and **q ∈ {0.5, 0.75}**:

    location_i = q-quantile of its 512 calibrated probabilities
    scale_i    = mean probability above location_i - location_i

The prior location and scale are equal-problem averages. The character prior
is the pooled mean output length. During four-fold policy selection these
priors exclude the validation fold; after selection they are refit on all 41
training problems.

### Online fit

At prefix **n**:

    local_location = q-quantile of opened probabilities
    tail            = opened probabilities >= local_location
    local_scale     = mean(tail - local_location)

    location = shrink local_location toward training location using n samples
    scale    = shrink local_scale toward training scale using tail_count samples

With confidence coefficient **alpha**, the policy uses

    radius       = sqrt(log(1 / 0.05) / tail_count)
    location_ucb = clip(location + alpha × scale × radius, 0, 1)
    scale_ucb    = scale × (1 + alpha × radius)

The fitted future tail is a shifted exponential truncated and renormalized on
**[location_ucb, 1]**, with fixed mass **1 - q**. Expected improvement is
evaluated with 32-point Gauss–Legendre quadrature.

The estimated next character count is

    (opened_character_sum + cost_prior_strength × training_mean_characters)
    /
    (n + cost_prior_strength).

## One-sentence descriptions of the saved coding comparisons

### results/coding/utility_gap

- **Shifted-exponential Pandora:** for each divisor, select the
  shifted-exponential configuration with maximum four-fold training utility,
  refit its prior on all 41 training problems, and evaluate its held-out
  accuracy minus exact characters divided by the divisor.
- **Train-tuned Fixed-N:** select the integer **N** maximizing exact
  without-replacement training accuracy minus **N × mean_characters / D**,
  then freeze that **N** on test.
- **Equal-character Fixed-N:** on independent test permutations, interpolate
  between adjacent integer **N** values so Fixed-N has exactly the adaptive
  policy’s aggregate expected character count; this is a descriptive
  held-out match, not a deployable selector.

The utility search uses **q ∈ {0.5, 0.75}**, confidence
**{0, 0.2, 0.8}**, reward-prior strength **{5, 20, 10^6}**, cost-prior
strength **{0, 10}**, no tail decay, and a horizon of either 512 or a selected
multiple of train-tuned Fixed-N. Ninety-one of 100 saved
shifted-exponential policies use a finite horizon.

Representative findings:

- At **D = 100,000**, relative utility gain is 5.75% (95% CI 2.81–8.69) and
  equal-character accuracy gain is 1.51 percentage points (0.55–2.47).
- At **D = 300,000**, relative utility gain is 1.85% (0.09–3.61) and
  equal-character accuracy gain is 0.73 points (0.17–1.28).
- Most higher-divisor utility intervals include zero, so the method is not
  uniformly better at every reported character price.

### results/coding/target_quality

- **Shifted-exponential target policy:** for each requested accuracy, select
  one uncapped shifted-exponential configuration and divisor that passes a
  10%-current-training/90%-frozen-development accuracy score at minimum frozen
  development-profile character cost.
- **Train-target Fixed-N:** choose the cheapest training integer **N** whose
  exact training accuracy reaches the requested target and freeze it on test.
- **Held-out matched Fixed-N:** choose post hoc the cheapest test integer **N**
  reaching the adaptive policy’s achieved test accuracy; this is a descriptive
  oracle, not a deployable baseline.

Target selection uses **q ∈ {0.5, 0.75}**, confidence **{0, 0.2, 0.8}**,
reward-prior strength **{5, 20, 10^6}**, cost-prior strength **{0, 10}**,
tail decay **{0, 0.25, 0.5, 1}**, 19 divisors, and the full 512-generation
horizon. Every retained selection has positive tail decay.

| Requested accuracy | Achieved accuracy | Mean characters | Saving vs train Fixed-N | Saving vs held-out matched Fixed-N | Per-split hit rate |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.2503 | 2,371 | 63.28% | 33.86% | 5/10 |
| 0.30 | 0.2923 | 6,768 | 76.38% | 28.06% | 3/10 |
| 0.35 | 0.3448 | 39,328 | 53.57% | 35.69% | 3/10 |

The mean quality is close to the requested target, but the method does not
guarantee target attainment on every held-out split. The 95% interval for the
0.35 saving against held-out matched Fixed-N includes zero.

## Claim verification

| Claim | Verdict |
|---|---|
| The coding rule is an unknown-distribution Pandora policy. | Correct: its tail law is estimated from training data and the observed prefix. |
| The opening cost is unknown. | Correct for the next output length; **D** itself is a selected unit conversion. |
| Pandora acts after a utility transform. | Correct: the retained coding transform is train-fitted isotonic correctness probability. |
| The calibration was fit on final held-out test problems. | Incorrect: it was fit on the 41 outer-training problems and frozen for the 42 final-test problems. |
| The calibration was fully nested within inner policy-selection folds. | Incorrect: the isotonic map is outer-train global, while the distribution prior is fold-specific. |
| Every retained coding policy uses UCB. | Incorrect: zero-confidence selections are plug-in Pandora policies. |
| The target experiment is pure classical Pandora. | Incorrect: every selected target policy also uses empirical tail decay. |
| The utility experiment is always uncapped. | Incorrect: 91 of 100 selected policies use a training-derived hard horizon. |

The most accurate umbrella name is:

> **Unknown-distribution, unknown-next-cost Pandora-style
> expected-improvement stopping with train-fitted utility calibration,
> distribution and character priors, optional UCB optimism, and
> experiment-specific stopping regularization.**
