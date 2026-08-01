"""Statistics for anomaly detection and localization.

Pure functions over plain numbers: no database, no configuration, no I/O. That is deliberate.
These are the calculations every published claim rests on, so they need to be testable against
hand-worked examples without standing anything up.

Four ideas here are worth stating outright, because getting any of them wrong produces a
system that is confidently and undetectably miscalibrated.

**A rate is not a mean of rates.** The expected fill rate over four historical weeks is
``sum(fills) / sum(requests)``, the volume-weighted maximum-likelihood estimate -- never the
average of four daily fill rates. The two agree only when every week carried identical
traffic, and the difference grows exactly when traffic is unstable, which is when anomalies
happen.

**Robustness with four samples comes from trimming one, not from a median.** A median of four
is the mean of the middle two, which is barely more robust than the mean. Dropping the single
most extreme week is a 25% trim, and it survives one contaminated week -- which this dataset
guarantees, since a planted global outage poisons every Sunday baseline.

**The dispersion estimate must divide by N-G, not N.** The group rate is estimated from the
same samples it is compared against, which deflates residuals by exactly m/(m-1). With four
samples per cell that understates dispersion by a third, and an understated dispersion makes
every test look better calibrated than it is.

**An estimated scale is not a known scale.** A spread measured from four days is itself a
random quantity, and when those four days happen to land close together it comes out far too
small. Reading such a statistic off a normal tail asserts the scale was known exactly, which
on this corpus turned a fall of 5% into p = 1.6e-155 and buried every fill-rate finding
underneath it. Student's t on k-1 degrees of freedom prices the uncertainty in and leaves the
same statistic worth the 1e-4 it is actually worth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Two-sided normal critical values, for readability at the call sites.
Z_ALPHA_01 = 2.5758293035489004  # alpha = 0.01
Z_POWER_80 = 0.8416212335729143  # power = 0.80


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of an empty sequence")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def mad(values: Sequence[float], *, scale: float = 1.4826) -> float:
    """Median absolute deviation, scaled to be comparable with a standard deviation."""
    if not values:
        return 0.0
    med = median(values)
    return scale * median([abs(v - med) for v in values])


def normal_sf(z: float) -> float:
    """Two-sided tail probability of the standard normal.

    ``erfc`` is used rather than ``1 - cdf`` because the latter loses all precision past about
    z = 6, and this system routinely produces larger z values on genuine incidents. Reporting
    p = 0 where the true value is 1e-40 is harmless; reporting p = 0 because the subtraction
    underflowed hides how strong the evidence actually was.
    """
    return math.erfc(abs(z) / math.sqrt(2.0))


#: Iteration cap and convergence tolerance for the incomplete-beta continued fraction. Swept
#: across one to a million degrees of freedom and statistics from 1e-4 to 1e8, the fraction
#: converges in four terms typically and forty at worst, so the cap bounds a pathological input
#: rather than the working case and returning the unconverged value on reaching it is not a
#: silent wrong answer waiting to happen.
_BETACF_MAX_ITER = 200
_BETACF_EPS = 3e-16

#: Substituted for a continued-fraction denominator that has collapsed toward zero. Lentz's
#: method divides by that denominator on the following line, and the recurrence genuinely
#: passes near zero for ordinary inputs, so the substitution is load-bearing rather than a
#: precaution against inputs that never arrive.
_BETACF_TINY = 1e-300


def _betacf(x: float, a: float, b: float) -> float:
    """Continued fraction for the incomplete beta function, by Lentz's method.

    Lentz is used rather than evaluating the fraction from the bottom up because the depth
    needed for a given accuracy is not known in advance; the bottom-up form has to be restarted
    at increasing depths and the results compared, which costs several times the arithmetic to
    reach the same answer.
    """

    def guard(value: float) -> float:
        return _BETACF_TINY if abs(value) < _BETACF_TINY else value

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 / guard(1.0 - qab * x / qap)
    h = d
    for m in range(1, _BETACF_MAX_ITER + 1):
        m2 = 2 * m
        even = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / guard(1.0 + even * d)
        c = guard(1.0 + even / c)
        h *= d * c
        odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / guard(1.0 + odd * d)
        c = guard(1.0 + odd / c)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPS:
            break
    return h


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function, I_x(a, b).

    The continued fraction converges quickly only while x stays below (a+1)/(a+b+2); beyond
    that point the reflection I_x(a, b) = 1 - I_{1-x}(b, a) moves the evaluation back onto the
    fast side. Both branches share one leading factor because x^a (1-x)^b is invariant under
    the same exchange.

    The leading factor is assembled in log space. Formed directly it is a ratio of gamma
    functions that overflows for perfectly ordinary degrees of freedom -- a t test on 350
    samples is already past the limit of a double -- while the logarithms stay near zero.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def student_t_sf(t: float, df: float) -> float:
    """Two-sided tail probability of Student's t on ``df`` degrees of freedom.

    The sign of the statistic is discarded, exactly as `normal_sf` discards it. Metrics fall at
    least as often as they rise, and callers rank on p-value without consulting the direction,
    so a one-sided tail here would score every drop at p near 1 and silently delete half of
    what this system exists to find.

    Reaching for t rather than the normal is what keeps a scale estimated from a handful of
    samples honest. A normal tail treats an estimated scale as though it were known exactly,
    and when four historical points happen to land close together the statistic explodes: this
    corpus returned p = 1.6e-155 for a move of 5% on exactly that arithmetic. Three degrees of
    freedom price the same statistic near 1e-4, which is the strength the evidence has.

    ``df <= 0`` means no second sample ever existed to estimate a scale from, so there is no
    distribution to consult and no evidence to report.
    """
    if not math.isfinite(df) or df <= 0.0:
        return 1.0
    # t * t rather than t ** 2, which raises OverflowError past about 1e154. Saturating to
    # infinity sends x to zero and `_betainc` reports an underflowed tail, which is the right
    # answer for a statistic that large; raising would take down a scan of thousands of
    # segments over one degenerate cell.
    return _betainc(df / (df + t * t), df / 2.0, 0.5)


def pooled_rate(samples: Sequence[tuple[float, float]]) -> float:
    """Volume-weighted rate over ``(denominator, numerator)`` pairs."""
    n = sum(s[0] for s in samples)
    k = sum(s[1] for s in samples)
    return k / n if n else float("nan")


@dataclass(frozen=True)
class Pooled:
    n: float
    k: float
    rate: float
    weeks_kept: int
    weeks_seen: int
    dropped_rate: float | None

    @property
    def usable(self) -> bool:
        return self.weeks_kept >= 1 and self.n > 0


def trim_and_pool(samples: Sequence[tuple[float, float]], *, trim: bool = True) -> Pooled:
    """Drop the single most extreme historical week, then pool the rest.

    ``samples`` are ``(denominator, numerator)`` pairs, one per historical week. Extremeness is
    judged on the rate, not the volume: a week with unusual traffic but a normal rate is
    perfectly good evidence about the rate, and discarding it would throw away the sample that
    best anchors a noisy segment.
    """
    usable = [(n, k) for n, k in samples if n and n > 0]
    seen = len(usable)
    if seen == 0:
        return Pooled(0.0, 0.0, float("nan"), 0, 0, None)
    if not trim or seen < 3:
        # Below three samples there is nothing to trim toward: dropping one of two leaves a
        # single observation, which is a baseline of one week pretending to be robust.
        n = sum(s[0] for s in usable)
        k = sum(s[1] for s in usable)
        return Pooled(n, k, k / n if n else float("nan"), seen, seen, None)

    rates = [k / n for n, k in usable]
    med = median(rates)
    worst = max(range(seen), key=lambda i: abs(rates[i] - med))
    kept = [s for i, s in enumerate(usable) if i != worst]
    n = sum(s[0] for s in kept)
    k = sum(s[1] for s in kept)
    return Pooled(n, k, k / n if n else float("nan"), len(kept), seen, rates[worst])


def pearson_dispersion(
    cells: Sequence[tuple[float, float, float]], n_groups: int
) -> float:
    """Overdispersion factor from ``(denominator, numerator, group_rate)`` cells.

    Divides by ``N - G`` rather than ``N``. Real ad traffic is never exactly binomial across
    segments -- there is always structure the model does not carry -- so a naive estimate that
    comes back below 1.0 is not evidence of under-dispersion, it is evidence of the bias.
    """
    residuals = pearson_residuals(cells)
    dof = len(residuals) - n_groups
    if dof <= 0:
        return 1.0
    return sum(residuals) / dof


def pearson_residuals(cells: Sequence[tuple[float, float, float]]) -> list[float]:
    """Squared standardised binomial residuals, one per usable cell."""
    out = []
    for n, k, p in cells:
        if n <= 0 or not (0.0 < p < 1.0):
            continue
        variance = n * p * (1.0 - p)
        if variance <= 0:
            continue
        out.append((k - n * p) ** 2 / variance)
    return out


def quasi_poisson_dispersion(
    cells: Sequence[tuple[float, float]], n_groups: int
) -> float:
    """Overdispersion for count metrics, from ``(observed, expected)`` cells.

    Request arrivals are driven by shared conditions rather than being independent, so counts
    are reliably overdispersed relative to Poisson -- often by a large factor. Assuming pure
    Poisson would treat ordinary traffic variation as overwhelming evidence.
    """
    residuals = poisson_residuals(cells)
    dof = len(residuals) - n_groups
    if dof <= 0:
        return 1.0
    return sum(residuals) / dof


def poisson_residuals(cells: Sequence[tuple[float, float]]) -> list[float]:
    """Squared standardised Poisson residuals, one per usable cell."""
    return [
        (observed - expected) ** 2 / expected for observed, expected in cells if expected > 0
    ]


#: Median of a chi-square distribution with one degree of freedom. Under the null, a squared
#: standardised residual follows that distribution, so dividing the median squared residual by
#: this constant gives an estimate on the same scale as the mean-based one.
_CHI2_1DF_MEDIAN = 0.454936423

#: Scales the median squared residual to the mean-based dispersion scale.
_ROBUST_SCALE = 1.0 / _CHI2_1DF_MEDIAN


def robust_dispersion(squared_residuals: Sequence[float]) -> float:
    """Overdispersion from the *median* squared residual rather than the mean.

    The mean-based estimators are sums of squared residuals divided by degrees of freedom, so a
    single contaminated observation dominates the total. That is not a hypothetical: on this
    corpus one baseline week carried a real, global fill-rate incident, and the mean-based
    estimate came back at 25 to 128 depending on the segment -- pegged against the ceiling of
    50. Every denominator floor is proportional to phi, so a fifty-fold inflation made every
    cell in the lattice untestable and the detector reported a clean bill of health for a day on
    which fill rate had visibly moved. The clamp that was supposed to prevent exactly this
    instead delivered it, because the cap is far above the point where detection dies.

    Taking the median tolerates up to half the observations being contaminated, which for four
    weekly samples means one bad week costs nothing. Under the null a squared standardised
    residual is chi-square with one degree of freedom, so the median is divided by that
    distribution's median to land on the same scale the mean-based estimator would give on clean
    data.

    Trimming the worst week instead would also work, but it discards a real observation on the
    basis of the very quantity being estimated, and with four samples that biases the result
    downward by construction. The median makes no such choice.
    """
    usable = [r for r in squared_residuals if math.isfinite(r) and r >= 0.0]
    if not usable:
        return 1.0
    return median(usable) * _ROBUST_SCALE


def clamp_dispersion(phi: float, floor: float = 1.0, ceiling: float = 50.0) -> float:
    """Keep the estimate inside a defensible range.

    Floored at 1.0 because a genuinely sub-binomial process would mean segments are negatively
    correlated, which ad serving is not; a sub-1 estimate is sampling noise or residual bias,
    and honouring it would make every test more confident than the data supports. Capped
    because one pathological segment can otherwise inflate phi until nothing is ever
    detectable, turning the detector silently off.
    """
    if not math.isfinite(phi):
        return floor
    return max(floor, min(ceiling, phi))


@dataclass(frozen=True)
class TestResult:
    # "Test" here means statistical hypothesis test, but the name matches pytest's collection
    # pattern, so importing it into a test module makes pytest try to collect it as a suite.
    __test__ = False

    z: float
    p_value: float
    observed: float
    expected: float
    absolute_effect: float
    relative_effect: float
    model: str

    @property
    def direction(self) -> str:
        if self.absolute_effect > 0:
            return "rise"
        if self.absolute_effect < 0:
            return "fall"
        return "flat"

    @property
    def testable(self) -> bool:
        """Whether a comparison was actually formed, as opposed to declined.

        Every test in this module returns ``p = 1`` and a zero effect when it cannot run -- no
        baseline exposure, too few usable weeks, an expectation of zero. Read naively that is
        indistinguishable from a confident finding of "this cell did not move", and asserting
        that about a cell nobody managed to measure is a worse error than staying quiet, because
        it will clear a real culprit.

        The shared signature of every such path is a NaN expectation: there was no value to
        compare against. Deriving the flag from that rather than from a list of model names
        means a test added later cannot forget to declare itself untestable.
        """
        return not math.isnan(self.expected)


def two_proportion_test(
    k_obs: float, n_obs: float, k_base: float, n_base: float, *, phi: float = 1.0
) -> TestResult:
    """Two-proportion z-test with pooled variance, inflated by the dispersion factor.

    The baseline arm carries real sampling error too, which is why this is a two-proportion
    test rather than a one-sample test against a point estimate. Treating four weeks of
    history as if it were the exact truth overstates significance on every thinly-trafficked
    segment -- precisely the segments most likely to be reported by mistake.
    """
    if n_obs <= 0 or n_base <= 0:
        return TestResult(0.0, 1.0, float("nan"), float("nan"), 0.0, 0.0, "two_proportion")

    p_obs = k_obs / n_obs
    p_base = k_base / n_base
    p_pool = (k_obs + k_base) / (n_obs + n_base)

    variance = phi * p_pool * (1.0 - p_pool) * (1.0 / n_obs + 1.0 / n_base)
    if variance <= 0:
        # A degenerate pooled rate (every request filled, or none did) has no binomial
        # variance to test against. Silence is the honest answer, not infinite confidence.
        return TestResult(0.0, 1.0, p_obs, p_base, p_obs - p_base, 0.0, "two_proportion")

    z = (p_obs - p_base) / math.sqrt(variance)
    return TestResult(
        z=z,
        p_value=normal_sf(z),
        observed=p_obs,
        expected=p_base,
        absolute_effect=p_obs - p_base,
        relative_effect=(p_obs / p_base - 1.0) if p_base else 0.0,
        model="two_proportion",
    )


def count_test(observed: float, expected: float, *, phi: float = 1.0) -> TestResult:
    """Quasi-Poisson test for count metrics.

    Variance is ``phi * expected`` rather than ``expected``: request counts are driven by
    shared traffic conditions, so arrivals are correlated and a pure Poisson assumption
    understates the noise by a wide margin at scale.
    """
    if expected <= 0:
        # NaN rather than the zero that was passed in, so the result reports itself as
        # untestable. A zero expectation carries no scale, so there is no size against which
        # the observation could be called normal or surprising.
        return TestResult(0.0, 1.0, observed, float("nan"), 0.0, 0.0, "quasi_poisson")
    variance = phi * expected
    z = (observed - expected) / math.sqrt(variance)
    return TestResult(
        z=z,
        p_value=normal_sf(z),
        observed=observed,
        expected=expected,
        absolute_effect=observed - expected,
        relative_effect=observed / expected - 1.0,
        model="quasi_poisson",
    )


#: Floor on the log-space scale, in units of relative change. A MAD over four samples is the
#: mean of the two middle absolute deviations, so two history days landing near the median drag
#: it toward zero however wide the underlying spread really is, and nothing in the arithmetic
#: below distinguishes that accident from a genuinely quiet segment. One percent is the point
#: below which a revenue-per-impression series is not plausibly stable but merely under-sampled:
#: this corpus produced four-day spreads near 0.2% on segments whose eCPM then moved 5%, and
#: scoring a routine move at twenty-six standard deviations is what buried every fill-rate
#: finding underneath it. The floor is applied after the degeneracy check rather than in place
#: of it, because a history that is flat to numerical precision carries no scale information at
#: all and handing it a fabricated one would manufacture findings out of fixed-price inventory.
_MIN_LOG_SPREAD = 0.01


def log_ratio_test(
    observed: float, history: Sequence[float], *, min_history: int = 3
) -> TestResult:
    """Robust test for a continuous ratio such as eCPM or revenue per request.

    These are not proportions, so no binomial variance model applies. The comparison is made in
    log space against the spread of the segment's own history, measured with a MAD so that one
    previously anomalous week does not widen the interval enough to hide the current one.

    Three things separate this from a z-test against that spread, and all three bite hard at
    the four samples of history this system actually has.

    The scale is estimated, never known. Handing an estimate from four points to a normal tail
    claims a precision nobody has, and when those four points happen to land close together the
    statistic explodes: on this corpus eCPM and revenue-per-request came back at p = 0 and
    p = 1.6e-155 for falls of 5%, which is not strong evidence but a broken test, and they
    crowded every fill-rate and CTR finding out of the published list. Student's t on k - 1
    degrees of freedom prices that uncertainty in and leaves the same statistic worth about
    1e-4.

    What is being predicted is a new observation rather than the centre itself, so the standard
    error is sigma * sqrt(1 + 1/k): the centre carries its own error of sigma/sqrt(k), and the
    new point varies by a further sigma around wherever the centre truly is. Dropping the term
    understates the error by 12% at four samples, which at these statistics is worth orders of
    magnitude in the tail.

    The scale is also floored, at `_MIN_LOG_SPREAD`, because heavier tails alone do not save a
    segment whose four days happened to land almost on top of one another. Nothing here can
    tell that accident apart from a genuinely quiet segment, and the floor bounds how confident
    it is allowed to make the test.

    A near-zero MAD means the history was flat to numerical precision. That is treated as
    untestable rather than as infinite confidence, because a flat history usually means a
    segment too small to have moved at all.
    """
    usable = [v for v in history if v and v > 0]
    if observed <= 0 or len(usable) < min_history:
        return TestResult(0.0, 1.0, observed, float("nan"), 0.0, 0.0, "log_ratio_insufficient")

    logs = [math.log(v) for v in usable]
    k = len(logs)
    centre = median(logs)
    expected = math.exp(centre)

    # `mad` applies the 1.4826 consistency factor, so this is an estimate of sigma rather than
    # the raw median deviation. The raw one understates the scale of normally distributed data
    # by a third and inflates every statistic in the same proportion.
    sigma = mad(logs)

    if sigma < 1e-9:
        return TestResult(0.0, 1.0, observed, expected, observed - expected,
                          observed / expected - 1.0, "log_ratio_degenerate")

    standard_error = max(sigma, _MIN_LOG_SPREAD) * math.sqrt(1.0 + 1.0 / k)
    t = (math.log(observed) - centre) / standard_error
    return TestResult(
        z=t,
        p_value=student_t_sf(t, k - 1),
        observed=observed,
        expected=expected,
        absolute_effect=observed - expected,
        relative_effect=observed / expected - 1.0,
        model="log_ratio_t",
    )


def required_denominator(
    baseline_rate: float,
    relative_effect: float,
    *,
    alpha_z: float = Z_ALPHA_01,
    power_z: float = Z_POWER_80,
    phi: float = 1.0,
) -> float:
    """Denominator needed to detect a given relative change in a proportion.

    This is what makes a single global volume floor indefensible. Fill rate sits near 0.79 and
    CTR near 0.02; at the same relative effect the CTR test needs roughly two orders of
    magnitude more denominator. One constant floor is therefore simultaneously far too lax for
    one metric and far too strict for another, and the lax side is the expensive one -- it
    produces confident findings on segments that never had the traffic to support them.
    """
    p1 = baseline_rate
    p2 = p1 * (1.0 - relative_effect)
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0) or p1 == p2:
        return float("inf")
    p_bar = (p1 + p2) / 2.0
    numerator = (
        alpha_z * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + power_z * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return phi * numerator / (p1 - p2) ** 2


#: Largest relative drop worth sizing a floor against. A cell that cannot resolve a fall of
#: this size cannot resolve anything, because the metric would have to more than collapse for
#: the test to fire. Not fitted to any dataset -- it is the edge of the possible, since a
#: relative drop of 1.0 takes the rate to zero and makes the required sample size undefined.
NEAR_TOTAL_COLLAPSE = 0.95


def resolvable_effect(
    baseline_rate: float,
    denominator: float,
    *,
    alpha_z: float = Z_ALPHA_01,
    power_z: float = Z_POWER_80,
    phi: float = 1.0,
    tolerance: float = 1e-4,
) -> float | None:
    """Smallest relative drop this much traffic can resolve, or None if none can be.

    The inverse of `required_denominator`, and the honest way to express a detection limit: it
    is a property of the cell in front of you, not a constant chosen in advance. Reporting it
    turns "this segment was not tested" into "this segment could only have shown a fall of 12%
    or more", which an operator can act on and a reviewer can check.

    Solved by bisection rather than algebraically. The closed form is a quartic in the effect
    once the pooled-variance term is expanded, and its roots need care near p2 -> 0; bisection
    over a monotone function costs about forty evaluations of arithmetic that is already cheap.
    """
    if denominator <= 0 or not 0.0 < baseline_rate < 1.0:
        return None

    def needed(effect: float) -> float:
        return required_denominator(
            baseline_rate, effect, alpha_z=alpha_z, power_z=power_z, phi=phi
        )

    # Monotone decreasing in effect, so if even a near-total collapse needs more traffic than
    # the cell has, nothing is resolvable and the cell is genuinely untestable.
    if needed(NEAR_TOTAL_COLLAPSE) > denominator:
        return None

    lo, hi = 0.0, NEAR_TOTAL_COLLAPSE
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if needed(mid) > denominator:
            lo = mid
        else:
            hi = mid
    return hi


def wilson_interval(k: float, n: float, *, z: float = Z_ALPHA_01) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the normal approximation because it stays inside [0, 1] and remains sane
    for small n and extreme rates, where the textbook interval produces bounds below zero and
    invites nonsense conclusions about tiny segments.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def benjamini_hochberg(
    p_values: Sequence[float], alpha: float = 0.01, *, tests: int | None = None
) -> list[bool]:
    """Benjamini-Hochberg step-up procedure, returning a keep/reject mask.

    A single scan tests thousands of segments, so an uncorrected 1% threshold would produce
    tens of confident findings from noise alone. Controlling the false discovery rate rather
    than the family-wise error rate is the right trade here: the cost of one spurious case in a
    published list is an operator's afternoon, not a wrong decision, and Bonferroni over
    thousands of correlated segments would suppress genuine incidents.

    ``tests`` is the size of the family, which is the number of hypotheses *tested* and not the
    number of p-values handed to this function. They differ whenever the caller has already
    dropped some tested cells -- rises when only falls are wanted, say. Understating it is the
    one way to make this procedure silently permissive, since every threshold is alpha*k/m and
    a smaller m raises all of them. Overstating it only costs power, so the parameter is
    clamped upward rather than trusted in both directions.
    """
    n = len(p_values)
    if n == 0:
        return []
    m = max(n, tests or 0)

    # Ranks come from the position within the full family. The supplied p-values are the
    # smallest n of m, so their ranks are 1..n regardless of how much larger m is; only the
    # threshold alpha*rank/m widens with the family.
    order = sorted(range(n), key=lambda i: p_values[i])
    keep = [False] * n
    largest = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / m:
            largest = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= largest:
            keep[idx] = True
    return keep
