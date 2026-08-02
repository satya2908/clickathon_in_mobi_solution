"""Localizing against siblings, when the history has been rejected.

The property under test is not "does it find the segment" in the abstract, but whether the
answer it gives is the *smallest true* one and whether it declines in the cases where a sibling
comparison does not mean anything. A localizer that always names something is worse than useless
when the baseline has already been thrown out.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from verdict.config import Config
from verdict.detect import Finding
from verdict.metrics import MetricRegistry
from verdict.query import Counters, Segment, Window
from verdict.schema import TOTAL_COMBO
from verdict.stats import TestResult
from verdict.structloc import MIN_LEVELS, SiblingLocalizer, at_rate, sibling_norm


@pytest.fixture
def cfg() -> Config:
    return Config.model_validate(
        {
            "clickhouse": {"host": "localhost", "database": "test"},
            "run": {"data_dir": "."},
            "llm": {"enabled": False},
        }
    )


@pytest.fixture
def registry() -> MetricRegistry:
    return MetricRegistry.load("config/metrics.yaml")


@pytest.fixture
def window() -> Window:
    return Window(start=datetime(2026, 7, 8), end=datetime(2026, 7, 9), grain="1h")


def cell(requests: int, rate: float, *, render: float = 0.98, ctr: float = 0.011) -> Counters:
    """A coherent funnel at a given fill rate."""
    fills = int(round(requests * rate))
    impressions = int(round(fills * render))
    clicks = int(round(impressions * ctr))
    return Counters(
        requests=requests,
        fills=fills,
        impressions=impressions,
        clicks=clicks,
        revenue=impressions * 0.0032,
    )


class LatticeReader:
    """Serves prepared one-way and two-way slices with no database behind them."""

    def __init__(self, slices: dict[str, dict[Segment, Counters]]) -> None:
        self.slices = slices
        self.asked: list[str] = []

    def slice(self, combo: str, window: Window) -> dict[Segment, Counters]:  # noqa: ARG002
        self.asked.append(combo)
        return self.slices.get(combo, {})


def one_way(dimension: str, rates: dict[str, tuple[int, float]]) -> dict[Segment, Counters]:
    return {
        Segment(((dimension, value),)): cell(requests, rate)
        for value, (requests, rate) in rates.items()
    }


def total_of(cells: dict[Segment, Counters]) -> dict[Segment, Counters]:
    out = Counters()
    for counters in cells.values():
        out = out + counters
    return {Segment.total(): out}


def finding_for(metric: str, window: Window) -> Finding:
    return Finding(
        metric=metric,
        segment=Segment.total(),
        window=window,
        detector="structural",
        test=TestResult(-8.0, 0.0, 0.6, 0.79, -0.19, -0.24, "median_polish_residual"),
        observed_counters=Counters(),
        baseline_counters=Counters(),
        phi=1.0,
        effect_threshold=0.05,
    )


#: One dimension where a single level has collapsed, and four others behaving normally.
BROKEN = {
    "iOS 17.5": (120_000, 0.478),
    "iOS 18.1": (90_000, 0.791),
    "iOS 17.2": (60_000, 0.788),
    "Android 15": (80_000, 0.793),
    "Android 14": (50_000, 0.786),
}


class TestAtRateKeepsTheFunnelCoherent:
    """The counterfactual counters are subtracted from each other by the sufficiency test, and
    `Counters.__sub__` collapses to empty on any incoherent result. A counterfactual that quietly
    breaks the funnel does not produce a wrong answer, it produces no answer at all."""

    def test_it_hits_the_target_rate(self, registry):
        metric = registry.metric("fill_rate")
        out = at_rate(cell(100_000, 0.50), metric, 0.79)
        assert out is not None
        assert out.value(metric) == pytest.approx(0.79, abs=1e-5)

    def test_the_denominator_is_left_alone(self, registry):
        """Raising the fill rate means more of the same requests filled, not more requests."""
        metric = registry.metric("fill_rate")
        original = cell(100_000, 0.50)
        out = at_rate(original, metric, 0.79)
        assert out.requests == original.requests

    def test_stages_below_the_numerator_come_along(self, registry):
        """A cell that should have filled more would have served more impressions too. Leaving
        impressions behind gives the counterfactual a render rate above one."""
        fill_rate = registry.metric("fill_rate")
        render = registry.metric("render_rate")
        original = cell(100_000, 0.50)
        out = at_rate(original, fill_rate, 0.79)
        assert out.coherent
        assert out.value(render) == pytest.approx(original.value(render), abs=1e-3)

    def test_a_click_rate_change_leaves_revenue_alone(self, registry):
        """Revenue is earned per impression, so it does not follow clicks."""
        ctr = registry.metric("ctr")
        original = cell(100_000, 0.50)
        out = at_rate(original, ctr, 0.02)
        assert out.revenue == original.revenue
        assert out.impressions == original.impressions

    def test_an_empty_cell_refuses(self, registry):
        assert at_rate(Counters(), registry.metric("fill_rate"), 0.79) is None


class TestTheNormSurvivesTheIncident:
    """The whole method rests on the norm being something the incident cannot capture."""

    def test_a_large_broken_level_does_not_become_the_norm(self, registry):
        """The broken level here is the single biggest slice of traffic. A traffic-weighted
        centre would be dragged toward it; that is why the median is unweighted."""
        metric = registry.metric("fill_rate")
        norm = sibling_norm(one_way("os_version", BROKEN), metric, floor=1_000)
        assert norm == pytest.approx(0.791, abs=0.005)

    def test_it_holds_even_when_the_broken_level_is_most_of_the_traffic(self, registry):
        metric = registry.metric("fill_rate")
        dominant = dict(BROKEN)
        dominant["iOS 17.5"] = (2_000_000, 0.478)
        norm = sibling_norm(one_way("os_version", dominant), metric, floor=1_000)
        assert norm == pytest.approx(0.791, abs=0.005)

    def test_thin_levels_are_left_out(self, registry):
        """A level with a hundred requests has a rate that is mostly noise."""
        metric = registry.metric("fill_rate")
        noisy = dict(BROKEN)
        noisy["iOS 12.0"] = (100, 0.02)
        norm = sibling_norm(one_way("os_version", noisy), metric, floor=1_000)
        assert norm == pytest.approx(0.791, abs=0.005)

    def test_too_few_comparable_levels_gives_no_norm(self, registry):
        metric = registry.metric("fill_rate")
        pair = {"iOS 17.5": (120_000, 0.478), "iOS 18.1": (90_000, 0.791)}
        assert sibling_norm(one_way("os_version", pair), metric, floor=1_000) is None
        assert MIN_LEVELS == 3


class TestItNamesTheCollapsedLevel:
    def _localize(self, cfg, registry, window, rates=None, splits=None):
        cells = one_way("os_version", rates or BROKEN)
        slices = {TOTAL_COMBO: total_of(cells), "os_version": cells}
        slices.update(splits or {})
        reader = LatticeReader(slices)
        localizer = SiblingLocalizer(reader, registry, cfg.localization, cfg.detection)
        return localizer.localize(finding_for("fill_rate", window), direction="drop")

    def test_the_accused_is_the_level_that_collapsed(self, cfg, registry, window):
        out = self._localize(cfg, registry, window)
        assert out.accused is not None
        assert out.accused.segment.as_dict() == {"os_version": "iOS 17.5"}

    def test_the_expected_value_is_the_sibling_median(self, cfg, registry, window):
        out = self._localize(cfg, registry, window)
        assert out.accused.expected_value == pytest.approx(0.791, abs=0.005)

    def test_removing_it_accounts_for_the_gap(self, cfg, registry, window):
        out = self._localize(cfg, registry, window)
        assert out.accused.sufficiency > 0.9

    def test_the_verdict_says_where_the_expectation_came_from(self, cfg, registry, window):
        """An operator reading a case has to be able to tell that this number is a median of
        siblings and not a historical baseline, because the two license different actions."""
        out = self._localize(cfg, registry, window)
        assert out.mode == "siblings"
        assert "siblings" in out.note and "baseline audit" in out.note

    def test_the_confirmatory_test_is_measured_in_this_window(self, cfg, registry, window):
        """The detector's own p-value came from a median-polish residual on another grid, which
        is not evidence about this claim."""
        out = self._localize(cfg, registry, window)
        assert out.accused_finding is not None
        assert out.accused_finding.detector == "siblings"
        assert out.accused_finding.screening == "sibling_pooled"
        assert out.accused_finding.window == window

    def test_a_healthy_dimension_produces_no_accusation(self, cfg, registry, window):
        flat = {name: (requests, 0.79) for name, (requests, _) in BROKEN.items()}
        assert self._localize(cfg, registry, window, rates=flat).accused is None

    def test_a_uniform_regression_is_invisible(self, cfg, registry, window):
        """The documented blind spot, pinned so it stays documented. When every level falls
        together the median falls with them and nothing stands out -- which is exactly the case
        the temporal detector exists to catch."""
        sunk = {name: (requests, 0.40) for name, (requests, _) in BROKEN.items()}
        assert self._localize(cfg, registry, window, rates=sunk).accused is None


class TestItRefusesWhereSiblingsMeanNothing:
    def test_a_count_metric_gets_no_verdict(self, cfg, registry, window):
        """Segments differ in size for legitimate reasons, so "this segment should be the size
        of the median segment" is not a claim about health."""
        cells = one_way("os_version", BROKEN)
        reader = LatticeReader({TOTAL_COMBO: total_of(cells), "os_version": cells})
        localizer = SiblingLocalizer(reader, registry, cfg.localization, cfg.detection)
        out = localizer.localize(finding_for("requests", window), direction="rise")
        assert out.accused is None
        assert out.mode == "no_sibling_norm"

    def test_the_refusal_explains_itself(self, cfg, registry, window):
        cells = one_way("os_version", BROKEN)
        reader = LatticeReader({TOTAL_COMBO: total_of(cells), "os_version": cells})
        localizer = SiblingLocalizer(reader, registry, cfg.localization, cfg.detection)
        out = localizer.localize(finding_for("requests", window), direction="rise")
        assert "count" in out.note and "size" in out.note

    def test_no_rollup_rows_is_not_an_accusation(self, cfg, registry, window):
        reader = LatticeReader({})
        localizer = SiblingLocalizer(reader, registry, cfg.localization, cfg.detection)
        out = localizer.localize(finding_for("fill_rate", window), direction="drop")
        assert out.accused is None

    def test_a_rise_is_not_blamed_for_a_drop(self, cfg, registry, window):
        risen = dict(BROKEN)
        risen["iOS 17.5"] = (120_000, 0.95)
        cells = one_way("os_version", risen)
        reader = LatticeReader({TOTAL_COMBO: total_of(cells), "os_version": cells})
        localizer = SiblingLocalizer(reader, registry, cfg.localization, cfg.detection)
        assert localizer.localize(finding_for("fill_rate", window), direction="drop").accused is None
