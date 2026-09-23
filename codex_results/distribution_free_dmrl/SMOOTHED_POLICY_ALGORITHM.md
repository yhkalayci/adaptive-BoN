# Practical stopping rule: exact algorithm and results

This is an experiment report, not a manuscript edit. The practical rule needs
no offline training and fits no parametric response distribution. It is inspired
by the DMRL/order-statistic analysis, but the current theorem does not establish
its performance. Hyperparameters were investigated on Alpaca/Mistral; the
chosen variant was frozen before its FSFairX evaluation.

## Inputs and fixed parameters

- A prompt, generation model, and either the FSFairX or Mistral reward model.
- Actual price p per recorded generation-length unit.
- A maximum of N=960 responses and an initial batch of four.
- Benchmark quantile q=0.99; maximum utility valued at one dollar.
- Cost-optimism coefficient beta=2. This is a development-selected constant,
  not a fitted parameter needed at deployment and not a certified confidence level.

After n responses, r_i is the observed reward and ell_i the paid response
length. Actual accounting always charges p times the sum of all ell_i.

## Step 1: estimate the upper tail using order statistics

Let M_n=max(r_1,...,r_n) and z_i=exp(r_i-M_n). This numerical rescaling makes
the largest z_i equal to one; it does not change any Bradley–Terry probability.
Sort z in descending order, let k=floor(n/2), and set

    xi_n = z_(k+1),
    m_n = (1/k) sum_{j=1}^k (z_(j)-xi_n).

Thus xi_n is the lower empirical median and m_n is the average excess of
the upper half above it. The implementation retains exactly k observations;
ties at the threshold have zero excess.

Use the DMRL-motivated estimate of the benchmark in these normalized coordinates:

    B_n = xi_n + m_n [1 + log((k/n)/(1-q))].

This is an empirical extrapolation, not a fitted exponential distribution and
not a statistical upper-confidence bound. A corresponding population inequality
under DMRL motivates the formula; plugging in sample quantities does not inherit
that inequality as a confidence guarantee.

## Step 2: estimate the benefit of another response

For the current estimated benchmark define h_n(z)=z/(z+B_n). In raw-reward
coordinates this is the Bradley–Terry map against benchmark M_n+log(B_n).
The current best has transformed value z=1. Set

    g_n = [h_n(1+m_n)-h_n(1)]/(n+1)
        = B_n*m_n / [(n+1)(1+B_n)(1+B_n+m_n)].

Interpretation: estimate the chance of another record by 1/(n+1), and estimate
the utility gain on that event by adding a mean residual to the incumbent.
The unconditional record identity for continuous iid samples does NOT make
1/(n+1) a valid conditional probability after observing the scores. This
replacement is a heuristic, as is the empirical benchmark estimate.

## Step 3: average the estimated gain scale over time

Record a_n=n*g_n. At every response count n>=4, calculate

    A_n = (1/(n-3)) sum_{t=4}^n a_t,
    G_hat_n = A_n/n.

We average n times the estimated gain, not the raw one-step gains, because
the rule already models gain as decreasing on a 1/n scale. Old values of a_t
are retained as computed at time t; earlier observations are not rescored
using a later benchmark.

## Step 4: estimate next-call cost with a variance adjustment

Let ell_bar_n be the observed mean length, s_n its sample standard deviation
(denominator n-1), and se_n=s_n/sqrt(n). Use

    c_hat_n = p*ell_bar_n / [1 + beta*se_n/ell_bar_n],  beta=2.

This positive optimistic estimate equals the empirical mean cost when observed
lengths are constant and approaches it as relative estimation uncertainty falls.
It is not a distribution-free lower confidence bound. It affects stopping only;
reported profit never uses this discounted cost in place of actual expenditure.

## Step 5: stop or continue

    Generate and pay for four responses.
    Repeat:
        Update the upper-tail statistics and estimated benchmark.
        Compute the one-step gain estimate and update its running scale average.
        Estimate next-call cost using the observed mean and standard error.
        If G_hat_n <= c_hat_n, or n=N:
            Return the highest-reward response observed so far.
        Otherwise generate, score, and pay for one additional response.

All decisions are prefix-measurable. No unobserved reward, full-cache quantile,
future length, held-out label, or offline prompt-level predictor enters stopping.

## Completed Alpaca results

Five generation models, 100 prompts each, eight replay orders, six prices,
and five overlapping train/test splits. The adaptive rule is the same for all
generators and prices. Training prompts select only the fixed-N comparator,
searching every N=1,...,960.

| Reward model | Previous rule profit gain | New rule profit gain | New matched-quality cost saving |
|---|---:|---:|---:|
| FSFairX | 4.68% | **5.55%** | **27.73%** |
| Mistral | 3.85% | **4.998%** | **25.15%** |

Both new-rule profit gains are positive in 29/30 generator-price means.
Against a test-oracle fixed count, the mean gains are 4.88% and 4.30%,
respectively. The two-reward-model results must not be interpreted as independent
replications: they score the same generated responses and prompts.

Removing cost optimism while retaining smoothing gives 5.26% for FSFairX and
4.62% for Mistral. Thus smoothing accounts for most of the measured improvement;
the cost adjustment provides a smaller additional gain.

Matched savings compare each price-based adaptive policy with the cheapest
retrospective fixed-count mixture attaining its measured mean quality. These
are not savings for a training-free controller guaranteed to hit a requested
quality target. The separate target-quality figure uses retrospective mixtures
on both sides and flags infeasible cases.

## Near-direct theoretical policy

At n=4,8,16,..., transform rewards using Bradley–Terry against the current
prefix's empirical 0.99 quantile. Let v1>=v2>=v3 be the largest utilities. Set

    m_hat_n = ((v1-v3)+(v2-v3))/2.

If v3>0, stop when 4*m_hat_n/n <= p*ell_bar_n; otherwise continue to the next
doubling checkpoint. All policies return at the finite cap. The sequential
ablation changes only the checkpoint schedule to every response from n=4.

| Policy | FSFairX profit change | Mistral profit change |
|---|---:|---:|
| Near-direct, doubling | −31.56% | −36.86% |
| Same statistic, every response | −13.26% | −16.48% |

The near-direct policy performs poorly at these finite prices. In the Mistral
highest-price condition, it averages roughly 57 responses versus 4 for fixed N,
and its mean profit becomes negative. Its optimism and batching are costly.
This is not a counterexample to the theorem: utilities are estimated with a
moving benchmark, costs are estimated, DMRL is unverified, sampling is from a
finite cache, and the theorem gives an asymptotic rather than finite-price
near-optimality claim.

The new practical method changes the tail width, gain formula, smoothing,
checkpoint schedule, and cost estimate. It should not be called an almost-direct
implementation of the theoretical algorithm.

## Generalization and verification

The best setting was chosen during repeated Alpaca/Mistral development; FSFairX
used the frozen setting. A separate HH-RLHF/Mistral check with beta=1—not the
beta=2 setting above—achieved 2.10%, versus 1.82% for the previous rule. The
unsuccessful three-sample-start variant achieved 0.91%. Thus the evidence does
not establish a universal 5% improvement or even 5% across alignment datasets.

All 56 experiment tests pass. Arithmetic and LP audits accompany the runs;
the FSFairX primary has 150 checked matched-quality comparisons. Plot CSVs
preserve all displayed aggregates. Current caches charge recorded character
lengths; no token-count rerun was performed. The manuscript remains unchanged.

Implementations: `online_smoothed_spacing.py` defines the gain and averaging;
`online_cost_optimism.py` defines the cost adjustment; `two_reward_models.py`
freezes the comparison; `near_direct_comparison.py` evaluates the near-direct rule.
