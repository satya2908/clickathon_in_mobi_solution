"""What a case carries with it, and what it leaves behind."""

from __future__ import annotations

from datetime import datetime

from verdict.config import Config
from verdict.detect import Finding
from verdict.localize import Candidate, Localization
from verdict.metrics import MetricRegistry
from verdict.pipeline import _finding_for_accused, detect_all, for_metric
from verdict.query import Counters, Segment, Window
from verdict.stats import TestResult
from verdict.trace import Step


class TestACaseCarriesItsOwnMetricsSweep:
    """A ten-metric run emits a detection span per metric per lattice combination. Copying all
    of them onto every case made four thousand rows for nine cases, most of it a fill_rate case
    carrying the CTR scan -- spans that are not evidence for that verdict."""

    def _steps(self) -> list[Step]:
        return [
            Step(step_id="1", parent_id="", ordinal=1, name="investigate", kind="pipeline"),
            Step(step_id="2", parent_id="1", ordinal=2, name="detect", kind="detector"),
            Step(step_id="3", parent_id="2", ordinal=3, name="temporal:fill_rate:__all__", kind="detector"),
            Step(step_id="4", parent_id="2", ordinal=4, name="temporal:ctr:region", kind="detector"),
            Step(step_id="5", parent_id="2", ordinal=5, name="structural:fill_rate:country|region", kind="detector"),
            Step(step_id="6", parent_id="1", ordinal=6, name="correct", kind="statistics"),
        ]

    def test_another_metrics_scan_is_left_out(self):
        kept = {s.name for s in for_metric(self._steps(), "fill_rate")}
        assert "temporal:ctr:region" not in kept

    def test_its_own_scan_is_kept(self):
        kept = {s.name for s in for_metric(self._steps(), "fill_rate")}
        assert "temporal:fill_rate:__all__" in kept
        assert "structural:fill_rate:country|region" in kept

    def test_the_stages_that_hold_the_tree_together_are_always_kept(self):
        """Dropping a stage with no metric in its name would orphan everything beneath it."""
        kept = {s.name for s in for_metric(self._steps(), "ctr")}
        assert {"investigate", "detect", "correct"} <= kept

    def test_every_kept_step_still_has_its_parent(self):
        for metric in ("fill_rate", "ctr", "revenue"):
            kept = for_metric(self._steps(), metric)
            ids = {s.step_id for s in kept}
            assert all(not s.parent_id or s.parent_id in ids for s in kept), metric

    def test_a_metric_that_never_ran_keeps_only_the_scaffolding(self):
        assert [s.name for s in for_metric(self._steps(), "revenue")] == [
            "investigate",
            "detect",
            "correct",
        ]


WINDOW = Window(datetime(2026, 7, 8), datetime(2026, 7, 9), "1h")
SEGMENT = Segment.of(country="US", region="NAM")


def _finding(metric: str, *, observed: float, expected: float, p_value: float = 1e-9) -> Finding:
    return Finding(
        metric=metric,
        segment=SEGMENT,
        window=WINDOW,
        detector="temporal",
        test=TestResult(
            z=-9.4,
            p_value=p_value,
            observed=observed,
            expected=expected,
            absolute_effect=observed - expected,
            relative_effect=(observed - expected) / expected,
            model="two_proportion",
        ),
        observed_counters=Counters(requests=37_045, fills=22_190),
        baseline_counters=Counters(requests=13_650, fills=10_700),
        phi=1.0,
        weeks_kept=3,
        weeks_seen=4,
    )


class TestACaseQuotesATestOfItsOwnMetric:
    """The case's name comes from the finding and its numbers come from the localization, so a
    finding for the wrong metric produces a case that contradicts itself.

    `everything` spans the whole sweep. A segment that moved in two metrics has a finding in
    each, and matching on segment alone let a fill_rate localization on country=US AND region=NAM
    quote the *requests* finding for that same cell. The case went to ClickHouse as
    `metric=requests, observed=0.599` -- a request count of 0.6, which is the kind of number the
    whole system exists to not produce.
    """

    def _localization(self) -> Localization:
        accused = Candidate(
            segment=SEGMENT,
            observed=Counters(requests=37_045, fills=22_190),
            expected=Counters(requests=13_650, fills=10_700),
            observed_value=0.59902,
            expected_value=0.78398,
            status="accused",
        )
        return Localization(
            metric="fill_rate",
            window=WINDOW,
            parent=Segment.total(),
            parent_observed=0.7314,
            parent_expected=0.7846,
            parent_deviation=-0.0532,
            accused=accused,
            candidates=[accused],
            mode="explain_away",
        )

    def test_another_metrics_finding_for_the_same_cell_is_not_quoted(self):
        """Nothing for fill_rate in the sweep, so the wrong-metric finding must not stand in for
        one. The localizer's own re-test of the accused cell is the fallback, and here there is
        none, so the honest answer is that no finding describes this claim."""
        loc = self._localization()
        requests = _finding("requests", observed=37_045, expected=13_650, p_value=1e-12)
        assert _finding_for_accused(loc, [], [requests]) is None

    def test_its_own_metrics_finding_is_quoted_when_the_sweep_tested_it(self):
        loc = self._localization()
        requests = _finding("requests", observed=37_045, expected=13_650, p_value=1e-12)
        fill_rate = _finding("fill_rate", observed=0.59902, expected=0.78398)
        chosen = _finding_for_accused(loc, [], [requests, fill_rate])
        assert chosen is fill_rate

    def test_the_group_is_still_preferred_over_the_wider_sweep(self):
        loc = self._localization()
        in_group = _finding("fill_rate", observed=0.59902, expected=0.78398, p_value=1e-4)
        elsewhere = _finding("fill_rate", observed=0.59902, expected=0.78398, p_value=1e-30)
        assert _finding_for_accused(loc, [in_group], [elsewhere]) is in_group


class LatticeReader:
    """Serves one combo's cells with history, so the sweep can run without a database."""

    def __init__(self, cells: dict[Segment, list[Counters]], combo: str = "region") -> None:
        self.cells = cells
        self.combo = combo

    def prefetch_lattice(self, combos, window, weeks):  # noqa: ARG002
        return 0

    def slice_with_history(self, combo, window, weeks):  # noqa: ARG002
        return self.cells if combo == self.combo else {}

    def slice(self, combo, window):  # noqa: ARG002
        return {seg: arms[0] for seg, arms in self.cells.items()} if combo == self.combo else {}


class TestTheSweepCountsWhatItTested:
    """The correction sizes its family from `tested_cells`, so losing that count makes
    Benjamini-Hochberg silently permissive: every threshold is alpha*k/m, and a smaller m
    raises all of them. `detect_all` used to copy findings and gaps across from each metric's
    result by hand and leave the count behind, so the family arriving at the correction was
    always zero and collapsed to the number of findings."""

    def _world(self) -> dict[Segment, list[Counters]]:
        # Five arms: the window, then four aligned baseline weeks.
        steady = [Counters(requests=10_000) for _ in range(5)]
        risen = [Counters(requests=20_000)] + [Counters(requests=10_000) for _ in range(4)]
        return {
            Segment((("region", "APAC"),)): list(steady),
            Segment((("region", "EU"),)): list(steady),
            Segment((("region", "NAM"),)): risen,
        }

    def _run(self, **overrides):
        cfg = Config.model_validate(
            {
                "clickhouse": {"host": "localhost", "database": "test"},
                "run": {"data_dir": "."},
                "llm": {"enabled": False},
            }
        )
        for key, value in overrides.items():
            setattr(cfg.detection, key, value)
        window = Window(start=datetime(2026, 7, 8), end=datetime(2026, 7, 9), grain="1h")
        return detect_all(
            LatticeReader(self._world()),
            MetricRegistry.load("config/metrics.yaml"),
            cfg,
            window,
            metrics=["requests"],
            structural=False,
        )

    def test_the_count_reaches_the_caller(self):
        temporal, _ = self._run()
        assert temporal.tested_cells > 0

    def test_it_counts_cells_rather_than_findings(self):
        """With rises ignored, most tested cells yield no finding. The family is still every
        cell that was tested, which is the case the lost count would have understated."""
        temporal, _ = self._run(detect_rises=False)
        assert temporal.tested_cells > len(temporal.findings)
