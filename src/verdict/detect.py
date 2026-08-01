"""Temporal detection: does a segment differ from its own recent past?

This catches main effects -- a segment that used to behave one way and now behaves another. It
is the detector everyone builds, and on its own it has two blind spots this system covers
elsewhere: it cannot see an anomaly that exists only at the intersection of two dimensions
without moving either one much, and it cannot see two segments moving in opposite directions
by offsetting amounts. Both are present in the hackathon dataset. See ``structural.py``.

Three things here are less obvious than they look.

**The denominator floor is per metric, not global.** A cell is only tested when it carries
enough traffic to have detected the effect being looked for. With one constant floor, a
threshold sized for fill rate near 0.79 is roughly 150 times too lax for CTR near 0.02, and
segments get confidently reported on evidence that could never have supported the claim.

**Untestable cells are recorded, not dropped.** A segment skipped for thin traffic goes into a
coverage ledger. Silence about a segment and evidence of its innocence are very different
claims, and a system that cannot tell them apart will eventually be believed about the wrong
one.

**Thousands of tests need a false-discovery correction.** Scanning 46 combos at a 1% threshold
produces tens of confident findings from noise alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import DetectionConfig
from .metrics import Metric, MetricRegistry
from .query import Counters, RollupReader, Segment, Window
from .schema import LATTICE_DEPTH, TOTAL_COMBO
from .stats import (
    Pooled,
    TestResult,
    benjamini_hochberg,
    clamp_dispersion,
    count_test,
    log_ratio_test,
    pearson_dispersion,
    quasi_poisson_dispersion,
    required_denominator,
    trim_and_pool,
    two_proportion_test,
)
from .trace import Tracer

log = logging.getLogger(__name__)


@dataclass
class Finding:
    """A segment whose behaviour the detector could not explain as normal variation."""

    metric: str
    segment: Segment
    window: Window
    detector: str
    test: TestResult
    observed_counters: Counters
    baseline_counters: Counters
    phi: float
    weeks_kept: int = 0
    weeks_seen: int = 0
    survives_correction: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return self.test.direction

    @property
    def relative_effect(self) -> float:
        return self.test.relative_effect

    @property
    def p_value(self) -> float:
        return self.test.p_value

    def describe(self) -> str:
        return (
            f"{self.metric} for {self.segment.label()} {self.direction}s "
            f"{abs(self.relative_effect) * 100:.1f}% "
            f"({self.test.expected:.6g} to {self.test.observed:.6g}), p={self.p_value:.2g}"
        )


@dataclass
class CoverageGap:
    """A cell that could not be tested, and why.

    Published alongside the findings. Without it, "we found nothing in EMEA" is ambiguous
    between having looked and having been unable to look.
    """

    metric: str
    segment: Segment
    denominator: float
    required: float
    reason: str


@dataclass
class DetectionResult:
    findings: list[Finding] = field(default_factory=list)
    gaps: list[CoverageGap] = field(default_factory=list)
    dispersion: dict[str, float] = field(default_factory=dict)
    tested_cells: int = 0
    # False until `apply_correction` has run. While false, `findings` is every tested cell, not
    # a list of anomalies, and reporting from it would publish the uncorrected scan.
    corrected: bool = False

    def extend(self, other: DetectionResult) -> None:
        self.findings.extend(other.findings)
        self.gaps.extend(other.gaps)
        self.dispersion.update(other.dispersion)
        self.tested_cells += other.tested_cells
        self.corrected = self.corrected and other.corrected


def estimate_dispersion(
    history: dict[Segment, list[Counters]], metric: Metric, cfg: DetectionConfig
) -> float:
    """Estimate the overdispersion factor from the historical arms only.

    Pooling across every segment of a combo at once is what makes this usable: each segment
    contributes only four weekly samples, but the estimate is formed from all segments
    together and so carries hundreds of degrees of freedom rather than three.

    The window under investigation is deliberately excluded. Including it would let a genuine
    incident inflate the very dispersion figure used to judge whether it is significant, and
    a large enough anomaly would then explain itself away as noise.
    """
    is_proportion = metric.is_proportion
    cells: list[Any] = []
    n_groups = 0

    for weeks in history.values():
        hist = weeks[1:]
        if metric.is_ratio:
            usable = [c for c in hist if c.denominator(metric) and c.denominator(metric) > 0]
        else:
            usable = [c for c in hist if c.requests > 0]
        if len(usable) < 3:
            continue

        if is_proportion:
            num = sum(c.numerator(metric) for c in usable)
            den = sum(c.denominator(metric) or 0.0 for c in usable)
            if den <= 0:
                continue
            p = num / den
            if not (0.0 < p < 1.0):
                continue
            n_groups += 1
            cells.extend((c.denominator(metric), c.numerator(metric), p) for c in usable)
        elif metric.kind == "count":
            values = [c.numerator(metric) for c in usable]
            mean = sum(values) / len(values)
            if mean <= 0:
                continue
            n_groups += 1
            cells.extend((v, mean) for v in values)
        else:
            # Continuous ratios have no count-based variance model; log_ratio_test measures
            # spread from the segment's own history instead.
            return 1.0

    if not cells:
        return 1.0

    raw = (
        pearson_dispersion(cells, n_groups)
        if is_proportion
        else quasi_poisson_dispersion(cells, n_groups)
    )
    return clamp_dispersion(raw, cfg.dispersion_floor, cfg.dispersion_ceiling)


def _test_segment(
    metric: Metric, observed: Counters, pooled: Pooled, history: list[Counters], phi: float
) -> TestResult:
    if metric.is_proportion:
        return two_proportion_test(
            observed.numerator(metric),
            observed.denominator(metric) or 0.0,
            pooled.k,
            pooled.n,
            phi=phi,
        )
    if metric.kind == "count":
        expected = pooled.k / pooled.weeks_kept if pooled.weeks_kept else 0.0
        return count_test(observed.numerator(metric), expected, phi=phi)
    values = [v for v in (c.value(metric) for c in history) if v is not None]
    obs = observed.value(metric)
    return log_ratio_test(obs if obs is not None else 0.0, values)


def _pool_history(metric: Metric, history: list[Counters], *, trim: bool) -> Pooled:
    if metric.is_ratio:
        samples = [
            (c.denominator(metric) or 0.0, c.numerator(metric))
            for c in history
            if (c.denominator(metric) or 0.0) > 0
        ]
    else:
        # For an additive metric the "denominator" is the number of comparable weeks, so each
        # week contributes one unit of exposure carrying its own count.
        samples = [(1.0, c.numerator(metric)) for c in history if c.requests > 0]
    return trim_and_pool(samples, trim=trim)


def detect_temporal(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: DetectionConfig,
    metric_name: str,
    window: Window,
    *,
    combos: list[str] | None = None,
    tracer: Tracer | None = None,
    correct: bool = True,
) -> DetectionResult:
    """Scan one metric across the lattice, comparing each cell against its own history.

    Pass ``correct=False`` when several metrics will be published together, then call
    ``apply_correction`` once over the pooled result. Until it is called, ``result.findings``
    holds every tested cell rather than a reportable list.
    """
    metric = registry.metric(metric_name)
    result = DetectionResult()

    targets = combos if combos is not None else _lattice_combos(registry, metric, window.grain)
    for combo in targets:
        result.extend(
            _detect_temporal_combo(reader, registry, cfg, metric, window, combo, tracer)
        )

    if correct:
        apply_correction(result, cfg)
    return result


def apply_correction(result: DetectionResult, cfg: DetectionConfig) -> DetectionResult:
    """Control the false discovery rate over a whole family, then filter to what is reportable.

    The family is every cell that was tested and could have produced a finding. On this
    dataset a single metric at hourly grain tests on the order of 1,700 cells, so at an
    uncorrected threshold of 0.01 roughly seventeen cells per metric cross it by chance alone.
    Reporting those as incidents is how an operator learns to ignore the system.

    Call this once over the pooled result of every metric and window that will be published
    together. Correcting each metric separately leaves the overall error rate multiplied by the
    number of metrics, which is the same mistake as correcting per combo, one level up.
    """
    if not result.findings:
        return result

    keep = benjamini_hochberg(
        [f.p_value for f in result.findings],
        cfg.p_value_threshold,
        tests=result.tested_cells,
    )
    for finding, survives in zip(result.findings, keep, strict=True):
        finding.survives_correction = survives

    result.findings = [
        f
        for f in result.findings
        if f.survives_correction and abs(f.relative_effect) >= cfg.min_relative_effect
    ]
    result.corrected = True
    return result


def _lattice_combos(registry: MetricRegistry, metric: Metric, grain: str = "1h") -> list[str]:
    """Combos this metric may legally be sliced by at this grain, total first.

    Two filters apply. Combos containing a dimension the metric cannot legally use are omitted
    entirely rather than tested and discarded, because those values are not merely noisy --
    their denominator is a different population. And combos deeper than the grain stores are
    omitted because the rows do not exist; asking for them would return empty cells that look
    like segments with no traffic rather than like a question the storage cannot answer.
    """
    legal = set(registry.valid_dimensions(metric))
    depth = LATTICE_DEPTH.get(grain, 2)

    combos = [TOTAL_COMBO]
    ordered = sorted(legal)
    if depth >= 1:
        combos.extend(ordered)
    if depth >= 2:
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                combos.append(f"{a}|{b}")
    return combos


def _detect_temporal_combo(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: DetectionConfig,
    metric: Metric,
    window: Window,
    combo: str,
    tracer: Tracer | None,
) -> DetectionResult:
    result = DetectionResult()
    history = reader.slice_with_history(combo, window, cfg.baseline_weeks)
    if not history:
        return result

    phi = estimate_dispersion(history, metric, cfg)
    result.dispersion[f"{metric.name}:{combo}"] = phi

    # Sized once per combo from the combo's own overall rate, so the floor reflects this
    # metric's noise rather than a constant chosen with a different metric in mind.
    floor = _denominator_floor(metric, history, cfg, phi)

    for segment, weeks in history.items():
        observed, hist = weeks[0], weeks[1:]

        den = (observed.denominator(metric) if metric.is_ratio else observed.requests) or 0.0
        if den < floor:
            result.gaps.append(
                CoverageGap(
                    metric=metric.name,
                    segment=segment,
                    denominator=den,
                    required=floor,
                    reason="below_detection_floor",
                )
            )
            continue

        pooled = _pool_history(metric, hist, trim=cfg.trim_extremes)
        if pooled.weeks_kept < cfg.min_baseline_samples:
            result.gaps.append(
                CoverageGap(
                    metric=metric.name,
                    segment=segment,
                    denominator=den,
                    required=floor,
                    reason=f"only_{pooled.weeks_kept}_baseline_weeks",
                )
            )
            continue

        result.tested_cells += 1
        test = _test_segment(metric, observed, pooled, hist, phi)

        # Every tested cell is kept here, significant or not. Filtering by p-value at this
        # point and correcting afterwards is what made the correction a no-op: Benjamini-
        # Hochberg keeps the largest k where p(k) <= alpha*k/m, and if every input already
        # satisfies p <= alpha then k = m and nothing is ever rejected. The correction has to
        # see the p-values that failed, because the count of tests is the whole input to it.
        #
        # The effect-size gate moves after correction for the same reason: effect size and
        # p-value are strongly correlated, so dropping small effects first would remove the
        # bulk of the large p-values and reshape the null distribution the correction assumes.
        if test.direction == "rise" and not cfg.detect_rises:
            continue

        result.findings.append(
            Finding(
                metric=metric.name,
                segment=segment,
                window=window,
                detector="temporal",
                test=test,
                observed_counters=observed,
                baseline_counters=_mean_counters(hist),
                phi=phi,
                weeks_kept=pooled.weeks_kept,
                weeks_seen=pooled.weeks_seen,
                notes={"combo": combo, "dropped_week_rate": pooled.dropped_rate},
            )
        )

    if tracer is not None and (result.findings or result.gaps):
        with tracer.span(f"temporal:{metric.name}:{combo}", kind="detector") as span:
            span.what(
                f"Compared every {combo} cell of {metric.label} in {window.label()} against "
                f"the same window in each of the previous {cfg.baseline_weeks} weeks."
            )
            span.why(
                "Weekly alignment holds weekday and hour-of-day constant, so ordinary "
                "seasonality cannot masquerade as a change."
            )
            span.result(
                f"{len(result.findings)} cell(s) deviated beyond "
                f"{cfg.min_relative_effect:.0%} at p<{cfg.p_value_threshold}; "
                f"{len(result.gaps)} untestable; dispersion {phi:.2f}x."
            )
            span.set("verdict.phi", phi)
            span.set("verdict.cells_tested", result.tested_cells)

    return result


def _denominator_floor(
    metric: Metric, history: dict[Segment, list[Counters]], cfg: DetectionConfig, phi: float
) -> float:
    """Traffic a cell needs before its result is worth believing.

    Derived from the metric's own baseline rate and the smallest effect worth reporting, so a
    low-rate metric such as CTR automatically demands far more traffic than a high-rate one
    such as fill rate. A single shared constant cannot serve both.
    """
    if not metric.is_proportion:
        # Counts and continuous ratios have no proportion to size against; require enough
        # exposure that a single event cannot dominate the cell.
        return 100.0

    total_num = 0.0
    total_den = 0.0
    for weeks in history.values():
        for counters in weeks[1:]:
            total_num += counters.numerator(metric)
            total_den += counters.denominator(metric) or 0.0
    if total_den <= 0:
        return float("inf")

    rate = total_num / total_den
    needed = required_denominator(
        rate, cfg.min_relative_effect, phi=phi, power_z=_power_z(cfg.target_power)
    )
    # The floor applies to one window, while `needed` is per arm of the comparison. The
    # baseline arm pools several weeks and is correspondingly better resolved, so requiring
    # the full per-arm figure of the observation window alone would be too strict.
    return needed if needed != float("inf") else float("inf")


def _power_z(power: float) -> float:
    """Normal quantile for a target power, by bisection.

    Small enough not to warrant a dependency on scipy, and keeping the dependency list short
    matters for an image that has to build reliably.
    """
    import math

    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * math.erfc(-mid / math.sqrt(2.0))
        if cdf < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _mean_counters(history: list[Counters]) -> Counters:
    usable = [c for c in history if c.requests > 0]
    if not usable:
        return Counters()
    n = len(usable)
    total = Counters()
    for c in usable:
        total = total + c
    return Counters(
        requests=int(total.requests / n),
        fills=int(total.fills / n),
        impressions=int(total.impressions / n),
        clicks=int(total.clicks / n),
        revenue=total.revenue / n,
    )
