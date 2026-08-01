"""Persistence: that the row a case produces matches the columns it is inserted into, and that
what the model did survives the trip into ClickHouse.

The alignment tests look trivial and are not. Every row builder here returns a bare list and is
inserted positionally, so adding a column in one place and forgetting the other does not raise --
it shifts every later value one position left and writes them into the wrong columns. A p-value
lands in `dispersion`, prose lands in `fingerprint`, and the insert succeeds.

The narration tests exist because the system's central claim is that a figure it cannot verify
never reaches a reader. That claim is only checkable if the failures are recorded, so a discarded
draft has to leave a trace naming the model and the exact literals it made up.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from verdict.config import DetectionConfig
from verdict.detect import Finding
from verdict.localize import Candidate, Localization
from verdict.metrics import MetricRegistry
from verdict.narrate import Narration
from verdict.query import Counters, Segment, Window
from verdict.stats import TestResult
from verdict.store import (
    CANDIDATE_COLUMNS,
    CASE_COLUMNS,
    STEP_COLUMNS,
    build_case,
    estimate_impact,
)

REGISTRY = MetricRegistry.load("config/metrics.yaml")
WINDOW = Window(start=datetime(2026, 6, 23), end=datetime(2026, 6, 25), grain="1h")
SEGMENT = Segment.of(os_version="Android 15")
DETECTION = DetectionConfig()


def counters(requests: int = 100_000, rate: float = 0.785) -> Counters:
    return Counters(requests=requests, fills=int(requests * rate))


def finding() -> Finding:
    return Finding(
        metric="fill_rate",
        segment=SEGMENT,
        window=WINDOW,
        detector="temporal",
        test=TestResult(-139.9, 1e-300, 0.4339, 0.78545, -0.3516, -0.448, "two_proportion"),
        observed_counters=counters(54_225, 0.4339),
        baseline_counters=counters(54_225, 0.78545),
        phi=1.4,
    )


def localization(accused: bool = True) -> Localization:
    candidate = Candidate(
        segment=SEGMENT,
        observed=counters(54_225, 0.4339),
        expected=counters(54_225, 0.78545),
        observed_value=0.4339,
        expected_value=0.78545,
    )
    candidate.status = "accused"
    return Localization(
        metric="fill_rate",
        window=WINDOW,
        parent=Segment.total(),
        parent_observed=0.751,
        parent_expected=0.78545,
        parent_deviation=-0.0344,
        accused=candidate if accused else None,
        candidates=[candidate],
    )


def case(narration: Narration | None = None):
    return build_case(
        finding(),
        localization(),
        run_id="run-1",
        narration=narration,
        detected_at=datetime(2026, 6, 25, tzinfo=UTC),
    )


class TestRowsLineUpWithColumns:
    """Positional inserts fail silently when the two drift apart."""

    def test_case_row_matches_case_columns(self):
        assert len(case().case_row()) == len(CASE_COLUMNS)

    def test_candidate_rows_match_candidate_columns(self):
        for row in case().candidate_rows():
            assert len(row) == len(CANDIDATE_COLUMNS)

    def test_step_rows_match_step_columns(self):
        built = case()
        built.steps = [
            {
                "step_id": "s1",
                "parent_id": "",
                "ordinal": 0,
                "name": "detect",
                "kind": "sql",
                "what": "Sweeping the lattice",
                "why": "Every cell has to be tested before any is accused",
                "result": "322 findings survived correction",
                "sql": "SELECT 1",
                "duration_ms": 42,
                "span_id": "abc",
            }
        ]
        for row in built.step_rows():
            assert len(row) == len(STEP_COLUMNS)

    def test_the_narration_columns_are_where_they_claim_to_be(self):
        """Guards against a shift that would still produce a correctly sized row."""
        built = case(Narration("prose", "llm", True, [], "gemini-flash-latest", 4416, 312, 10037))
        row = dict(zip(CASE_COLUMNS, built.case_row(), strict=True))
        assert row["narrative_model"] == "gemini-flash-latest"
        assert row["narrative_verified"] == 1
        assert row["narrative_prompt_tokens"] == 4416
        assert row["narrative_completion_tokens"] == 312
        assert row["narrative_latency_ms"] == 10037


class TestWhatTheModelDidIsRecorded:
    def test_an_accepted_draft_records_the_model_that_wrote_it(self):
        built = case(Narration("prose", "llm", True, [], "gemini-flash-latest", 100, 20, 900))
        assert built.narrative_source == "llm"
        assert built.narrative_model == "gemini-flash-latest"
        assert built.narrative_verified is True
        assert built.narrative_rejected == []

    def test_a_discarded_draft_names_the_figures_it_invented(self):
        """The guardrail firing. Without this the fallback is indistinguishable from no model."""
        built = case(Narration("template prose", "template", False, ["6", "91%"], "gemini", 100, 40, 800))
        assert built.narrative_source == "template"
        assert built.narrative_verified is False
        assert built.narrative_rejected == ["6", "91%"]
        assert built.narrative_model == "gemini"

    def test_a_discarded_draft_still_records_what_it_cost(self):
        built = case(Narration("t", "template", False, ["6"], "gemini", 4416, 312, 10037))
        row = dict(zip(CASE_COLUMNS, built.case_row(), strict=True))
        assert row["narrative_prompt_tokens"] == 4416
        assert row["narrative_latency_ms"] == 10037

    def test_the_model_being_switched_off_is_distinguishable_from_it_failing(self):
        """No narration at all leaves the model blank, so 'never asked' reads differently from
        'asked and rejected'. Conflating them would make the rejection rate meaningless."""
        built = case(None)
        assert built.narrative_model == ""
        assert built.narrative_rejected == []
        assert built.narrative_source == "template"

    def test_rejected_figures_are_stored_as_strings(self):
        """They are literals as written, not values: "45%" and "0.45" are different evidence."""
        built = case(Narration("t", "template", False, ["45%"], "m", 1, 1, 1))
        row = dict(zip(CASE_COLUMNS, built.case_row(), strict=True))
        assert row["narrative_rejected"] == ["45%"]
        assert all(isinstance(f, str) for f in row["narrative_rejected"])


class TestImpactIsSignedLikeTheMovement:
    """A loss is negative and a recovery is positive.

    This was inverted: impact was stored as a shortfall, positive when the metric fell. Every
    reader treated it as a delta, so a segment that lost 113 clicks was rendered as ``+113``
    beside a downward arrow, and ``revenue at risk`` -- which sums ``min(0, revenue)`` so a
    recovery cannot cancel a breakage -- summed only zeroes and reported no money at risk.
    """

    def test_a_fall_is_negative(self):
        # 60 clicks observed against 173 expected: 113 clicks lost.
        impact = estimate_impact("ctr", Counters(impressions=1000), 0.060, 0.173)
        assert impact.units < 0
        assert impact.units == pytest.approx(-113.0)
        assert "short of expectation" in impact.basis[0]

    def test_a_rise_is_positive(self):
        impact = estimate_impact("ctr", Counters(impressions=1000), 0.173, 0.060)
        assert impact.units > 0
        assert "above expectation" in impact.basis[0]

    def test_money_lost_is_negative_so_revenue_at_risk_can_sum_it(self):
        # Fill rate ten points below expectation, on a segment that renders and earns.
        impact = estimate_impact(
            "fill_rate",
            Counters(requests=10_000, fills=6_000, impressions=6_000, revenue=60.0),
            0.60,
            0.70,
        )
        assert impact.units == pytest.approx(-1000.0)
        assert impact.revenue is not None
        assert impact.revenue < 0
        # min(0, revenue) is how the console keeps a recovery from cancelling a breakage; a
        # positive figure here would silently contribute nothing.
        assert min(0.0, impact.revenue) == impact.revenue

    def test_the_sign_agrees_with_the_direction_the_case_reports(self):
        for observed, expected, direction in ((0.05, 0.10, "fall"), (0.10, 0.05, "rise")):
            impact = estimate_impact("ctr", Counters(impressions=500), observed, expected)
            assert (impact.units < 0) == (direction == "fall")
