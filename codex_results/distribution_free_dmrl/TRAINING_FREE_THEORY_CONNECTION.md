# What is and is not justified for the training-free odds policy

This note separates population inequalities from the implemented plug-in
policy. It is not a new performance theorem and does not modify the manuscript.

## A population upper bound using mean residual life

Let R be a nonnegative random score with finite mean and decreasing mean
residual life m(s)=E[R-s | R>s]. Let the fixed monetary proxy be

    h_B(r) = r/(r+B),  B>0.

For exponentiated reward scores this is the Bradley–Terry transformation.
Suppose the incumbent is a >= xi, and set

    p = P(R>xi),  m = E[R-xi | R>xi],  d = a-xi.

For p>0 and m>0, the population expected improvement satisfies

    E[(h_B(R)-h_B(a))+]
      <= p * min(1, exp(1-d/m)) * B*m / ((a+B)*(a+B+m)).

Proof:

1. The residual X=R-xi conditioned on R>xi is DMRL. Its integrated tail
   L_X(t)=E[(X-t)+] obeys L_X(t)<=m*exp(-t/m), as established by integrating
   L_X'(t)=-L_X(t)/m_X(t) and using m_X(t)<=m.
2. If d>=m, P(X>d)<=L_X(d-m)/m<=exp(1-d/m). For d<m, use the bound 1.
   Multiplying by p bounds P(R>a).
3. Conditional on R>a, DMRL gives E[R-a | R>a]<=m. The function h_B is
   increasing and concave, so Jensen bounds the conditional utility gain by
   h_B(a+m)-h_B(a)=B*m/((a+B)*(a+B+m)).
4. Multiply the probability and conditional-gain bounds.

No exponential distribution is assumed. The exponential expression in the
bound follows from DMRL. Nonetheless, it resembles an exponential-tail plug-in
algebraically; that similarity should be acknowledged rather than concealed.

## Reference quantile bound

For q in (0,1), if p>=1-q, the population value

    b_q = xi + m*(1 + log(p/(1-q)))

satisfies P(R>b_q)<=1-q by step 2 above. This motivates the policy's online
0.99-quantile estimate. It is a population upper bound using true p and m,
not a confidence bound after replacing these quantities by empirical estimates.

## Connection to the implementation

`training_free_odds.py` uses the lower empirical median as xi, the fraction of
observations above it as p, and their mean excess as m. Dividing all odds by
the current maximum makes a=1 without changing Bradley–Terry utilities.

- The `shape_jensen_bound` variant directly substitutes these estimates into
  the displayed population gain bound and estimates B by the quantile formula.
- The prespecified primary `rank_jensen_bound` instead substitutes 1/(n+1)
  for P(R>a), resembling the 1/n factor in the manuscript's order-statistic rule.
  For continuous iid samples, 1/(n+1) is the unconditional chance that the next
  sample is a record. It is **not** a conditional upper confidence bound given
  the observed values.
- The empirical-gain variants average the observed residual utility increments
  instead of using Jensen. No fitted probability family is used, but this
  replacement is a heuristic rather than a certified population inequality.
- The `moment` reference variants omit the +1 in the quantile formula. That
  choice uses an exponential-tail moment approximation and loses the stated
  population quantile-upper-bound interpretation.

All variants compare estimated improvement with actual price times the prompt's
running mean response length. None learns a stopping price from other prompts.
Target-quality stopping compares estimated incumbent utility with the requested
target directly; it has no target-specific training step.

## Remaining gaps—do not claim the theorem covers the policy

1. DMRL is assumed for odds R here, not for the paper's utility variable. It has
   not been established for the experimental reward scores.
2. Empirical p and m are not confidence bounds; the median threshold is itself
   random and selected from the same observations.
3. B is estimated online. The gain inequality above assumes a fixed, known B.
   An upper estimate of B is not automatically conservative for utility gain:
   that gain is not monotone in B over its entire domain.
4. Every-response checking and running-mean random costs are not covered by
   the fixed-cost, doubling-checkpoint theorem.
5. The target-quality estimate need not be calibrated. Report attained quality
   and match cost comparisons to it, rather than assuming the target was met.
6. Cached replay is without replacement; the population argument assumes iid
   draws. No approximation ratio or confidence guarantee is asserted for these
   experimental substitutions.

Thus the connection is explicit and mathematically motivated, but structural,
not an inherited end-to-end guarantee. Both performance and the cost of closing
these proof gaps matter when choosing a final algorithm.

## Retry: remove the unknown benchmark from the stopping bound

`training_free_lipschitz.py` instead uses a raw reward R, which need not be
nonnegative. Fix a threshold xi and an incumbent a >= xi. Suppose the
conditional residual X=(R-xi | R>xi) has finite mean m>0 and is DMRL.
Let p=P(R>xi), and let h_b(r)=sigmoid(r-b) for any fixed benchmark b.
Then

    E[(h_b(R)-h_b(a))+] <= (p*m/4) * exp(-(a-xi)/m).

Indeed, h_b is increasing and globally 1/4-Lipschitz, independently of b.
Therefore its gain is at most (R-a)+/4, whose expectation is
p*L_X(a-xi)/4. To bound L_X, note that L_X is absolutely continuous and,
almost everywhere before its endpoint,

    L_X'(t) = -P(X>t) = -L_X(t)/m_X(t) <= -L_X(t)/m.

Integration gives L_X(t)<=m*exp(-t/m); beyond a finite endpoint L_X=0.
This proof also accommodates atoms, using derivatives almost everywhere.

This population inequality removes the need to know or estimate b for the
stopping statistic. The implementation uses empirical tail fractions and
residual means at an upper order statistic, then compares this expression
with price times the observed mean length after every response. No parametric
law, calibration set, or learned price is used. The `include` variants estimate
the mean from all observations above the threshold; `exclude` variants omit
the current maximum as an explicitly exploratory bias/variance alternative.

The advantages are a short derivation and no moving utility benchmark in the
decision. The limitations remain substantial: DMRL is imposed on raw reward
residuals, empirical estimates at data-dependent thresholds are not confidence
bounds, and the global Lipschitz constant may greatly overestimate utility
improvement. An unknown-mean cost estimate, repeated checking, and finite-cache
sampling still require separate analysis. The retry results should not be
presented as a theorem-backed end-to-end algorithm.
