"""Structural detection: does a cell differ from what its own row and column imply?

The temporal detector asks whether a segment changed against its past. This one asks a
different question entirely, inside a single window with no history at all: given how APAC
behaves overall and how iOS 18.1 behaves overall, is APAC x iOS 18.1 where it ought to be?

That distinction buys three things nothing else here provides.

**Interactions become visible.** An incident living only at the intersection of two dimensions
barely moves either one alone. Both marginals stay under any sensible threshold, so a
one-dimensional scan cannot see it at any confidence level. As a residual from the additive
fit, it is enormous.

**Compensating pairs stop hiding.** When one ad format's eCPM falls and another's rises by an
offsetting amount, every aggregate stays flat and a detector watching totals sees a healthy
system. The additive fit does not care that the total is unchanged; both cells are far from
where the row and column effects put them.

**It works with no baseline.** On a fresh slice with one week of data and no comparable
history, the temporal detector degrades badly and this one does not, because the comparison is
against siblings rather than against the past.

The method is Tukey's median polish on log values: fit ``log(v) = grand + row + column`` by
repeatedly sweeping out medians, then judge what is left over. Medians rather than means
because the outliers being searched for would otherwise pull the fit toward themselves and
partly erase the residual that identifies them. Logs because these are ratios, where a
proportional shift is the meaningful one.

The measured failure mode of this detector, taken seriously here: with one shared volume
threshold it produced zero false positives on eCPM and 141 on CTR across the same clean
windows. That is not a flaw in the polish, it is a floor sized for the wrong metric. Cells are
now admitted using the same per-metric power calculation the temporal detector uses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import combinations

from .config import DetectionConfig
from .detect import CoverageGap, DetectionResult, Finding
from .metrics import Metric, MetricRegistry
from .query import Counters, RollupReader, Segment, Window
from .schema import LATTICE_DEPTH
from .stats import (
    NEAR_TOTAL_COLLAPSE,
    TestResult,
    mad,
    median,
    normal_sf,
    required_denominator,
)
from .trace import Tracer

log = logging.getLogger(__name__)


def log_sampling_error(metric: Metric, counters: Counters) -> float | None:
    """Standard error of ``log(metric value)`` for one cell, from the sampling model.

    This is the yardstick each residual is judged against, and using it rather than a single
    spread shared by the whole grid matters for two reasons.

    A thin cell is naturally noisier than a busy one. Judging both against one number
    systematically over-flags small cells, which is most of what a shared threshold does
    wrong.

    A clean grid has almost every residual at exactly zero, so a spread measured from the
    residuals themselves collapses and the detector goes blind precisely when the background
    is most cooperative. A model-based error does not depend on the anomaly being surrounded
    by noise.

    Derived by the delta method: ``Var(log p) ~ Var(p)/p^2``.
    """
    if metric.is_proportion:
        n = counters.denominator(metric) or 0.0
        k = counters.numerator(metric)
        if n <= 0 or k <= 0:
            return None
        p = k / n
        if not (0.0 < p < 1.0):
            return None
        return math.sqrt((1.0 - p) / (n * p))

    if metric.kind == "count":
        n = counters.numerator(metric)
        return 1.0 / math.sqrt(n) if n > 0 else None

    # A continuous ratio such as eCPM has no count-based variance model: revenue per
    # impression has a spread of its own that this ignores. Treated as a lower bound, with
    # the overdispersion factor below picking up whatever it understates.
    den = counters.denominator(metric) or 0.0
    return 1.0 / math.sqrt(den) if den > 0 else None


@dataclass
class PolishResult:
    grand: float
    row_effects: dict[str, float]
    col_effects: dict[str, float]
    residuals: dict[tuple[str, str], float]
    iterations: int
    converged: bool


def median_polish(
    grid: dict[tuple[str, str], float], *, max_iterations: int = 20, tolerance: float = 1e-9
) -> PolishResult:
    """Tukey's median polish over a possibly incomplete grid.

    Sweeps row medians and column medians out of the table alternately until what remains
    stops changing. Missing cells are simply absent from the medians rather than imputed:
    inventing a value for an empty cell would invent the very structure the fit is meant to
    measure.
    """
    rows = sorted({r for r, _ in grid})
    cols = sorted({c for _, c in grid})
    residual = dict(grid)
    row_eff = dict.fromkeys(rows, 0.0)
    col_eff = dict.fromkeys(cols, 0.0)
    grand = 0.0
    converged = False
    iterations = 0

    for sweep in range(1, max_iterations + 1):
        iterations = sweep
        moved = 0.0

        for r in rows:
            present = [residual[(r, c)] for c in cols if (r, c) in residual]
            if not present:
                continue
            delta = median(present)
            if delta:
                for c in cols:
                    if (r, c) in residual:
                        residual[(r, c)] -= delta
                row_eff[r] += delta
                moved += abs(delta)

        delta = median(list(row_eff.values()))
        if delta:
            for r in rows:
                row_eff[r] -= delta
            grand += delta
            moved += abs(delta)

        for c in cols:
            present = [residual[(r, c)] for r in rows if (r, c) in residual]
            if not present:
                continue
            delta = median(present)
            if delta:
                for r in rows:
                    if (r, c) in residual:
                        residual[(r, c)] -= delta
                col_eff[c] += delta
                moved += abs(delta)

        delta = median(list(col_eff.values()))
        if delta:
            for c in cols:
                col_eff[c] -= delta
            grand += delta
            moved += abs(delta)

        if moved < tolerance:
            converged = True
            break

    return PolishResult(grand, row_eff, col_eff, residual, iterations, converged)


def detect_structural(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: DetectionConfig,
    metric_name: str,
    window: Window,
    *,
    tracer: Tracer | None = None,
) -> DetectionResult:
    """Scan every legal pair of dimensions for cells that break the additive pattern."""
    metric = registry.metric(metric_name)
    result = DetectionResult()

    if window.duration.days < cfg.structural_min_window_days:
        log.debug(
            "structural scan skipped for %s: window shorter than %d day(s)",
            metric_name,
            cfg.structural_min_window_days,
        )
        return result

    # The whole method is a two-way grid, so a grain that stores only one-way cells has nothing
    # for it to read. Returning empty here is honest; scanning anyway would build grids of zeros
    # and report every cell as a structural anomaly against a background of nothing.
    if LATTICE_DEPTH.get(window.grain, 2) < 2:
        log.debug(
            "structural scan skipped for %s: grain %s stores one-way cells only",
            metric_name,
            window.grain,
        )
        return result

    legal = registry.valid_dimensions(metric)
    for dim_a, dim_b in combinations(sorted(legal), 2):
        result.extend(
            _scan_pair(reader, cfg, metric, window, dim_a, dim_b, tracer)
        )
    return result


def _scan_pair(
    reader: RollupReader,
    cfg: DetectionConfig,
    metric: Metric,
    window: Window,
    dim_a: str,
    dim_b: str,
    tracer: Tracer | None,
) -> DetectionResult:
    result = DetectionResult()
    combo = "|".join(sorted((dim_a, dim_b)))
    cells = reader.slice(combo, window)
    if not cells:
        return result

    floor = cell_floor(metric, cells, cfg)

    grid: dict[tuple[str, str], float] = {}
    counters_by_cell: dict[tuple[str, str], Counters] = {}
    first, second = sorted((dim_a, dim_b))

    for segment, counters in cells.items():
        keys = segment.as_dict()
        value = counters.value(metric)
        den = (counters.denominator(metric) if metric.is_ratio else counters.requests) or 0.0

        if value is None or value <= 0:
            # Log space needs a positive value. A zero-rate cell is a real observation but
            # not one this model can represent, so it is recorded rather than dropped.
            result.gaps.append(
                CoverageGap(metric.name, segment, den, floor, "non_positive_value")
            )
            continue
        if den < floor:
            result.gaps.append(
                CoverageGap(metric.name, segment, den, floor, "below_detection_floor")
            )
            continue

        key = (keys.get(first, ""), keys.get(second, ""))
        grid[key] = math.log(value)
        counters_by_cell[key] = counters

    rows = {r for r, _ in grid}
    cols = {c for _, c in grid}
    if len(rows) < cfg.structural_min_levels or len(cols) < cfg.structural_min_levels:
        # Row and column effects have to be estimated from the same table the residuals come
        # from. Below three levels a single outlier is absorbed into the effect it should be
        # standing out against, and the residual it leaves is meaningless.
        return result

    polish = median_polish(grid)
    residuals = list(polish.residuals.values())

    errors = {k: log_sampling_error(metric, counters_by_cell[k]) for k in polish.residuals}
    usable_errors = [e for e in errors.values() if e and e > 0]
    if not usable_errors:
        result.gaps.append(
            CoverageGap(metric.name, Segment.of(**{first: "*", second: "*"}), 0.0, 0.0,
                        "no_sampling_model")
        )
        return result

    # How much wider the table actually scatters than pure sampling error predicts. Real ad
    # traffic always carries structure the row-and-column model does not, so this is floored
    # at 1 and never allowed to make the test more confident than the sampling model alone.
    spread = mad(residuals)
    typical_error = median(usable_errors)
    overdispersion = max(1.0, spread / typical_error) if typical_error > 0 else 1.0

    result.tested_cells += len(grid)
    effect_floor = metric.effect_threshold(cfg.min_relative_effect)

    for key, residual in polish.residuals.items():
        error = errors.get(key)
        if not error or error <= 0:
            continue
        z = residual / (error * overdispersion)
        effect = math.exp(residual) - 1.0
        if abs(z) < cfg.structural_z_threshold or abs(effect) < effect_floor:
            continue
        if effect > 0 and not cfg.detect_rises:
            continue

        counters = counters_by_cell[key]
        observed = counters.value(metric) or 0.0
        expected = observed / (1.0 + effect) if effect != -1.0 else 0.0
        segment = Segment(tuple(sorted(((first, key[0]), (second, key[1])))))

        result.findings.append(
            Finding(
                metric=metric.name,
                segment=segment,
                window=window,
                detector="structural",
                test=TestResult(
                    z=z,
                    p_value=normal_sf(z),
                    observed=observed,
                    expected=expected,
                    absolute_effect=observed - expected,
                    relative_effect=effect,
                    model="median_polish_residual",
                ),
                observed_counters=counters,
                baseline_counters=Counters(),
                phi=1.0,
                effect_threshold=effect_floor,
                # Screened by a fixed z threshold, not by Benjamini-Hochberg. These findings
                # never enter the temporal family -- pooling a median-polish residual with a
                # weekly-baseline p-value would mix two different nulls -- so they must not
                # claim a correction they did not go through.
                screening="structural_z",
                notes={
                    "combo": combo,
                    "residual_log": residual,
                    "sampling_error": error,
                    "overdispersion": overdispersion,
                    "residual_mad": spread,
                    "row_effect": polish.row_effects.get(key[0], 0.0),
                    "col_effect": polish.col_effects.get(key[1], 0.0),
                    "grid_shape": f"{len(rows)}x{len(cols)}",
                    "polish_converged": polish.converged,
                },
            )
        )

    if tracer is not None and result.findings:
        with tracer.span(f"structural:{metric.name}:{combo}", kind="detector") as span:
            span.what(
                f"Fitted an additive row-and-column model to {metric.label} across the "
                f"{len(rows)}x{len(cols)} {first} by {second} grid, using median polish on "
                "log values, and measured what each cell had left over."
            )
            span.why(
                "An anomaly confined to one intersection barely moves either dimension on "
                "its own, and a pair moving in opposite directions leaves every total "
                "unchanged. Neither is visible to a scan that compares against the past."
            )
            span.result(
                f"{len(result.findings)} cell(s) beyond {cfg.structural_z_threshold} standard "
                f"errors, with the table scattering {overdispersion:.1f}x wider than pure "
                "sampling error predicts."
            )
    return result


def cell_floor(metric: Metric, cells: dict[Segment, Counters], cfg: DetectionConfig) -> float:
    """Traffic a cell needs before its residual is worth reading.

    Sized from this metric's own overall rate. Applying one shared constant is what produced
    141 false positives on CTR while producing none on eCPM over identical clean windows: the
    threshold was adequate for a metric near 0.79 and roughly 150 times too lax for one near
    0.02.

    Sized against a near-total collapse rather than against ``min_relative_effect``, for the
    reasons ``detect._denominator_floor`` sets out at length and which apply identically here:
    a floor built from the smallest effect anyone hopes to see is only as good as a threshold
    chosen in advance, and choosing it well means fitting it to the corpus in hand.

    On this corpus the difference decided whether the metric existed. At CTR's 1.09% base rate
    a 5% floor demands roughly 827,000 impressions; a whole day has 184,000 across every cell
    combined, so every structural CTR cell was filed as below_detection_floor and the detector
    that exists to find interaction-only incidents never tested one. Against a 95% collapse the
    same cells need about 1,300, and the false-discovery correction rather than a volume cutoff
    is what keeps the noisy ones out of the published list.
    """
    if not metric.is_proportion:
        return 100.0

    num = sum(c.numerator(metric) for c in cells.values())
    den = sum(c.denominator(metric) or 0.0 for c in cells.values())
    if den <= 0:
        return float("inf")
    return required_denominator(num / den, NEAR_TOTAL_COLLAPSE)
