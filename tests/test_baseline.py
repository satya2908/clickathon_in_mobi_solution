"""The baseline audit: does history still describe the population?

The property under test is not "does it find incidents" -- that is the detector's job -- but
"does it tell the difference between an incident and a baseline that has stopped applying".
Those look identical in a single cell and are far apart in aggregate, which is the whole basis
of the check.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from verdict.baseline import BaselineAudit, audit_baseline
from verdict.config import Config
from verdict.metrics import MetricRegistry
from verdict.query import Window


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


class FakeReader:
    """Returns a fixed flagged share, and records which windows were asked about."""

    def __init__(self, shares: dict[str, tuple[int, int]] | None = None, default=(5, 1000)):
        self.shares = shares or {}
        self.default = default
        self.asked: list[str] = []

    def prefetch_lattice(self, combos, window, weeks):
        return 0


def _audit_with(monkeypatch, rates: list[tuple[int, int]], cfg, registry, window, **over):
    """Drive audit_baseline with canned per-window (survivors, tested) pairs."""
    calls: list[Window] = []

    def fake_share(reader, reg, config, win, names):
        calls.append(win)
        return rates[len(calls) - 1] if len(calls) <= len(rates) else (0, 0)

    monkeypatch.setattr("verdict.baseline.flagged_share", fake_share)
    for key, value in over.items():
        setattr(cfg.detection, key, value)
    return audit_baseline(FakeReader(), registry, cfg, window), calls


@pytest.fixture
def window() -> Window:
    return Window(
        start=datetime(2026, 7, 8), end=datetime(2026, 7, 9), grain="1h"
    )


class TestACalibratedBaselineIsAccepted:
    def test_a_few_percent_flagged_is_what_a_working_detector_looks_like(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(monkeypatch, [(11, 1000), (9, 1000)], cfg, registry, window)
        assert audit.trustworthy
        assert audit.flagged_rate == pytest.approx(0.009)

    def test_it_audits_the_windows_immediately_before_the_one_under_test(
        self, monkeypatch, cfg, registry, window
    ):
        _, calls = _audit_with(monkeypatch, [(1, 100), (1, 100)], cfg, registry, window)
        assert [c.start for c in calls][:2] == [datetime(2026, 7, 7), datetime(2026, 7, 6)]

    def test_the_window_under_test_is_read_too_but_kept_apart(
        self, monkeypatch, cfg, registry, window
    ):
        """It is read last and scored separately.

        Averaging it into the historical rate would be the worst of both: an incident inside the
        window would drag the historical reading up and switch the detector off on the day it is
        needed, which is the failure the minimum exists to prevent.
        """
        audit, calls = _audit_with(
            monkeypatch, [(1, 100), (1, 100), (400, 1000)], cfg, registry, window
        )
        assert calls[-1].start == window.start
        assert audit.flagged_rate == pytest.approx(0.01)
        assert audit.target_rate == pytest.approx(0.40)


class TestAnUnusableBaselineIsRejected:
    def test_flagging_most_of_the_grid_is_miscalibration_not_discovery(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(monkeypatch, [(430, 1000), (420, 1000)], cfg, registry, window)
        assert not audit.trustworthy
        assert audit.flagged_rate == pytest.approx(0.42)

    def test_the_explanation_says_what_will_happen_instead(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(monkeypatch, [(430, 1000)], cfg, registry, window)
        assert "structural" in audit.detail.lower()
        assert "43.0%" in audit.detail


class TestARealIncidentIsNotMistakenForMiscalibration:
    """The distinction the minimum exists to draw.

    An incident inflates the flagged share of the window it happened in. A dead baseline
    inflates every window at once. Taking the best of the recent windows means one bad day
    cannot switch the detector off -- which would be exactly backwards, since a bad day is when
    it is needed.
    """

    def test_one_disastrous_window_beside_a_calm_one_keeps_the_baseline(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(monkeypatch, [(600, 1000), (8, 1000)], cfg, registry, window)
        assert audit.trustworthy
        assert audit.flagged_rate == pytest.approx(0.008)

    def test_every_window_bad_rejects(self, monkeypatch, cfg, registry, window):
        audit, _ = _audit_with(monkeypatch, [(600, 1000), (550, 1000)], cfg, registry, window)
        assert not audit.trustworthy


class TestACorpusBoundaryIsCaughtOnTheDayItArrives:
    """The case the historical reading structurally cannot see.

    A corpus that reissues its dimension tables between one day and the next leaves every prior
    window clean -- they are all on the old side of the seam and agree with each other -- while
    every comparison across it is meaningless. Auditing only the past means staying silent for
    as many days as the audit is wide, and those are the days whose verdicts are worth least.
    """

    def test_clean_history_and_a_wrecked_window_rejects(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(
            monkeypatch, [(2, 1000), (2, 1000), (424, 1000)], cfg, registry, window
        )
        assert not audit.trustworthy
        assert audit.reason == "boundary"

    def test_it_reports_both_readings_so_the_shape_is_visible(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(
            monkeypatch, [(2, 1000), (2, 1000), (424, 1000)], cfg, registry, window
        )
        assert "42.4%" in audit.headline
        assert "0.2%" in audit.headline

    def test_the_explanation_names_the_boundary_rather_than_a_dead_baseline(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(
            monkeypatch, [(2, 1000), (2, 1000), (424, 1000)], cfg, registry, window
        )
        assert "boundary" in audit.detail.lower()

    def test_it_says_the_aggregate_survives(self, monkeypatch, cfg, registry, window):
        """Rejection is only safe because it is not blindness, so the explanation must say so."""
        audit, _ = _audit_with(
            monkeypatch, [(2, 1000), (2, 1000), (424, 1000)], cfg, registry, window
        )
        assert "aggregate" in audit.detail.lower()
        assert "sibling" in audit.detail.lower()

    def test_a_quiet_window_after_quiet_history_is_left_alone(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(
            monkeypatch, [(2, 1000), (3, 1000), (9, 1000)], cfg, registry, window
        )
        assert audit.trustworthy
        assert audit.reason == ""

    def test_a_baseline_wrong_for_a_while_is_named_as_that_not_as_a_boundary(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(
            monkeypatch, [(400, 1000), (410, 1000), (420, 1000)], cfg, registry, window
        )
        assert not audit.trustworthy
        assert audit.reason == "history"


class TestAnAuditThatCouldNotRun:
    def test_no_history_reports_that_it_did_not_check_rather_than_that_it_passed(
        self, monkeypatch, cfg, registry, window
    ):
        audit, _ = _audit_with(monkeypatch, [(0, 0), (0, 0)], cfg, registry, window)
        assert not audit.ran
        assert "not audited" in audit.headline.lower()

    def test_an_unchecked_baseline_still_permits_the_run(
        self, monkeypatch, cfg, registry, window
    ):
        # Refusing to investigate at the start of a corpus would be worse than proceeding: there
        # is no evidence the baseline is bad, only no evidence that it is good.
        audit, _ = _audit_with(monkeypatch, [(0, 0)], cfg, registry, window)
        assert audit.trustworthy


class TestTheBarIsConfigurable:
    def test_a_stricter_bar_rejects_what_the_default_accepts(
        self, monkeypatch, cfg, registry, window
    ):
        loose, _ = _audit_with(monkeypatch, [(80, 1000)], cfg, registry, window)
        assert loose.trustworthy

        strict, _ = _audit_with(
            monkeypatch, [(80, 1000)], cfg, registry, window, baseline_audit_max_flagged=0.05
        )
        assert not strict.trustworthy

    def test_the_number_of_audited_windows_is_configurable(
        self, monkeypatch, cfg, registry, window
    ):
        _, calls = _audit_with(
            monkeypatch, [(1, 100)] * 5, cfg, registry, window, baseline_audit_windows=4
        )
        # Four historical windows, then the window under test.
        assert len(calls) == 5
        assert calls[-1].start == window.start


class TestTheAuditIsReportable:
    def test_a_passing_audit_says_so_without_alarming_anyone(self):
        audit = BaselineAudit(trustworthy=True, flagged_rate=0.011, bar=0.10)
        assert "calibrated" in audit.headline.lower()
        assert audit.detail == audit.headline
