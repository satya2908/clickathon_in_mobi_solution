"""Statistics for anomaly detection and localization.

Pure functions over plain numbers: no database, no configuration, no I/O. That is deliberate.
These are the calculations every published claim rests on, so they need to be testable against
hand-worked examples without standing anything up.

Three ideas here are worth stating outright, because getting any of them wrong produces a
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
    total = 0.0
    used = 0
    for n, k, p in cells:
        if n <= 0 or not (0.0 < p < 1.0):
            continue
        variance = n * p * (1.0 - p)
        if variance <= 0:
            continue
        total += (k - n * p) ** 2 / variance
        used += 1
    dof = used - n_groups
    if dof <= 0:
        return 1.0
    return total / dof


def quasi_poisson_dispersion(
    cells: Sequence[tuple[float, float]], n_groups: int
) -> float:
    """Overdispersion for count metrics, from ``(observed, expected)`` cells.

    Request arrivals are driven by shared conditions rather than being independent, so counts
    are reliably overdispersed relative to Poisson -- often by a large factor. Assuming pure
    Poisson would treat ordinary traffic variation as overwhelming evidence.
    """
    total = 0.0
    used = 0
    for observed, expected in cells:
        if expected <= 0:
            continue
        total += (observed - expected) ** 2 / expected
        used += 1
    dof = used - n_groups
    if dof <= 0:
        return 1.0
    return total / dof


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
        return TestResult(0.0, 1.0, observed, expected, 0.0, 0.0, "quasi_poisson")
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


def log_ratio_test(
    observed: float, history: Sequence[float], *, min_history: int = 3
) -> TestResult:
    """Robust test for a continuous ratio such as eCPM or revenue per request.

    These are not proportions, so no binomial variance model applies. The comparison is made in
    log space against the spread of the segment's own history, measured with a MAD so that one
    previously anomalous week does not widen the interval enough to hide the current one.

    A near-zero MAD means the history was flat to numerical precision. That is treated as
    untestable rather than as infinite confidence, because a flat history usually means a
    segment too small to have moved at all.
    """
    usable = [v for v in history if v and v > 0]
    if observed <= 0 or len(usable) < min_history:
        return TestResult(0.0, 1.0, observed, float("nan"), 0.0, 0.0, "log_ratio_insufficient")

    logs = [math.log(v) for v in usable]
    centre = median(logs)
    spread = mad(logs)
    expected = math.exp(centre)

    if spread < 1e-9:
        return TestResult(0.0, 1.0, observed, expected, observed - expected,
                          observed / expected - 1.0, "log_ratio_degenerate")

    z = (math.log(observed) - centre) / spread
    return TestResult(
        z=z,
        p_value=normal_sf(z),
        observed=observed,
        expected=expected,
        absolute_effect=observed - expected,
        relative_effect=observed / expected - 1.0,
        model="log_ratio",
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


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.01) -> list[bool]:
    """Benjamini-Hochberg step-up procedure, returning a keep/reject mask.

    A single scan tests thousands of segments, so an uncorrected 1% threshold would produce
    tens of confident findings from noise alone. Controlling the false discovery rate rather
    than the family-wise error rate is the right trade here: the cost of one spurious case in a
    published list is an operator's afternoon, not a wrong decision, and Bonferroni over
    thousands of correlated segments would suppress genuine incidents.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    keep = [False] * m
    largest = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / m:
            largest = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= largest:
            keep[idx] = True
    return keep
