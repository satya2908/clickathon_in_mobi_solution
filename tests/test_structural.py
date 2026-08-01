"""Tests for median polish, the fit the structural detector reads residuals from."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from verdict.metrics import MetricRegistry
from verdict.query import Counters
from verdict.stats import mad
from verdict.structural import log_sampling_error, median_polish

METRICS_PATH = Path(__file__).resolve().parents[1] / "config" / "metrics.yaml"


def _additive_grid(rows: dict[str, float], cols: dict[str, float], grand: float = 1.0):
    return {(r, c): grand + rv + cv for r, rv in rows.items() for c, cv in cols.items()}


class TestMedianPolish:
    def test_recovers_a_purely_additive_table(self):
        """With no anomaly present every residual must be zero, or the detector would
        manufacture findings from its own fitting error."""
        grid = _additive_grid({"a": 0.0, "b": 0.5, "c": -0.5}, {"x": 0.0, "y": 0.2, "z": -0.3})
        result = median_polish(grid)
        assert all(abs(v) < 1e-9 for v in result.residuals.values())
        assert result.converged

    def test_isolates_a_single_contaminated_cell(self):
        """The whole premise: one cell moves, and the fit attributes the movement to that cell
        rather than smearing it across the row and column it sits in."""
        grid = _additive_grid({"a": 0.0, "b": 0.5, "c": -0.5}, {"x": 0.0, "y": 0.2, "z": -0.3})
        grid[("b", "y")] -= 1.0
        result = median_polish(grid)

        assert result.residuals[("b", "y")] == pytest.approx(-1.0, abs=0.01)
        others = [v for k, v in result.residuals.items() if k != ("b", "y")]
        assert max(abs(v) for v in others) < 0.01

    def test_survives_the_outlier_pulling_the_fit(self):
        """A mean-based fit would absorb part of a large outlier into its own row and column
        effects, shrinking the residual that identifies it. Medians do not."""
        grid = _additive_grid(
            {"a": 0.0, "b": 0.5, "c": -0.5, "d": 0.1}, {"w": 0.0, "x": 0.2, "y": -0.3, "z": 0.4}
        )
        grid[("c", "y")] += 5.0
        result = median_polish(grid)
        assert result.residuals[("c", "y")] == pytest.approx(5.0, abs=0.05)

    def test_detects_a_compensating_pair_that_leaves_totals_flat(self):
        """The case no aggregate watcher can see.

        One cell falls and another rises by the same amount, so every marginal and the grand
        total are unchanged. Both must still surface as residuals.
        """
        rows = {"EU": 0.0, "NAM": 0.3, "APAC": -0.2, "MEA": 0.1}
        cols = {"banner": 0.0, "native": 0.25, "interstitial": -0.15, "video": 0.05}
        grid = _additive_grid(rows, cols)

        before_total = sum(grid.values())
        grid[("EU", "interstitial")] -= 0.8
        grid[("EU", "native")] += 0.8
        assert sum(grid.values()) == pytest.approx(before_total)

        result = median_polish(grid)
        residuals = result.residuals
        assert residuals[("EU", "interstitial")] < 0 < residuals[("EU", "native")]

        # Both anomalies must stand clear of every innocent cell, which is what lets any
        # sensible scale separate them. The totals being unchanged is irrelevant to the fit.
        guilty = {("EU", "interstitial"), ("EU", "native")}
        worst_innocent = max(abs(v) for k, v in residuals.items() if k not in guilty)
        assert min(abs(residuals[k]) for k in guilty) > 10 * max(worst_innocent, 1e-6)

    def test_detects_an_interaction_invisible_in_both_marginals(self):
        """An incident at one intersection barely moves either dimension alone.

        Here a single cell of a 5x5 grid drops by 30%. Each marginal moves by roughly a fifth
        of that, under any threshold worth setting, while the residual is unmistakable.
        """
        rows = dict.fromkeys(["APAC", "EU", "NAM", "LATAM", "MEA"], 0.0)
        cols = dict.fromkeys(["iOS 18.1", "iOS 17.2", "Android 15", "Android 14", "Android 13"], 0.0)
        grid = {k: math.log(0.785) for k in _additive_grid(rows, cols)}

        drop = math.log(0.55) - math.log(0.785)
        grid[("APAC", "iOS 18.1")] += drop

        row_mean = sum(
            math.exp(v) for (r, _), v in grid.items() if r == "APAC"
        ) / 5
        assert abs(row_mean / 0.785 - 1) < 0.07  # marginal moves under 7%

        result = median_polish(grid)
        assert result.residuals[("APAC", "iOS 18.1")] == pytest.approx(drop, abs=0.01)
        worst_innocent = max(
            abs(v) for k, v in result.residuals.items() if k != ("APAC", "iOS 18.1")
        )
        assert worst_innocent < 0.01

    def test_handles_missing_cells_without_inventing_them(self):
        grid = _additive_grid({"a": 0.0, "b": 0.5, "c": -0.5}, {"x": 0.0, "y": 0.2, "z": -0.3})
        del grid[("a", "z")]
        result = median_polish(grid)
        assert ("a", "z") not in result.residuals
        assert all(abs(v) < 1e-6 for v in result.residuals.values())

    def test_uniform_table_has_no_residual_spread(self):
        """The degeneracy guard's trigger: nothing to measure deviation against."""
        grid = {(r, c): 2.0 for r in "abc" for c in "xyz"}
        result = median_polish(grid)
        assert mad(list(result.residuals.values())) == 0.0

    def test_terminates_on_a_pathological_table(self):
        grid = {(r, c): float((ord(r) * 7 + ord(c) * 13) % 5) for r in "abcde" for c in "vwxyz"}
        result = median_polish(grid, max_iterations=20)
        assert result.iterations <= 20


class TestSamplingScale:
    """The yardstick each residual is judged against.

    Using one spread shared by the whole grid fails in two directions at once: it treats a
    thin cell as strictly as a busy one, and it collapses to zero on a clean grid, blinding
    the detector exactly when the background is most cooperative.
    """

    def _metric(self, name: str):
        return MetricRegistry.load(METRICS_PATH).metric(name)

    def test_thin_cells_get_a_wider_tolerance_than_busy_ones(self):
        metric = self._metric("fill_rate")
        thin = log_sampling_error(metric, Counters(requests=500, fills=395))
        busy = log_sampling_error(metric, Counters(requests=500_000, fills=395_000))
        assert thin > busy
        # Sampling error falls as 1/sqrt(n): a thousand times the traffic, about a
        # thirty-second of the noise.
        assert thin / busy == pytest.approx(math.sqrt(1000), rel=0.01)

    def test_low_rate_metrics_are_noisier_at_equal_denominator(self):
        """Why CTR needs so much more traffic than fill rate to say anything."""
        registry = MetricRegistry.load(METRICS_PATH)
        fill = log_sampling_error(
            registry.metric("fill_rate"), Counters(requests=10_000, fills=7_850)
        )
        ctr = log_sampling_error(
            registry.metric("ctr"), Counters(impressions=10_000, clicks=200)
        )
        assert ctr > fill * 5

    def test_clean_grid_still_yields_a_usable_scale(self):
        """The regression that motivated this. On an almost perfectly additive grid nearly
        every residual is exactly zero, so a residual-derived spread is zero and the anomaly
        divides by nothing. A sampling-model scale is unaffected."""
        metric = self._metric("fill_rate")
        counters = Counters(requests=100_000, fills=78_500)
        error = log_sampling_error(metric, counters)
        assert error > 0

        grid = _additive_grid(
            dict.fromkeys("abc", 0.0), dict.fromkeys("xyz", 0.0), grand=math.log(0.785)
        )
        grid[("b", "y")] += math.log(0.70) - math.log(0.785)
        result = median_polish(grid)
        assert mad(list(result.residuals.values())) == 0.0  # the collapsing scale

        overdispersion = max(1.0, 0.0 / error)
        z = result.residuals[("b", "y")] / (error * overdispersion)
        assert abs(z) > 5

    def test_degenerate_inputs_return_no_scale(self):
        metric = self._metric("fill_rate")
        assert log_sampling_error(metric, Counters(requests=0, fills=0)) is None
        # Every request filled: no binomial variance, so nothing to measure against.
        assert log_sampling_error(metric, Counters(requests=100, fills=100)) is None
