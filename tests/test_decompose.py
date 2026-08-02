"""Rate versus mix: the distinction that decides whether an aggregate move is a fault."""

from __future__ import annotations

from pathlib import Path

import pytest

from verdict.decompose import rate_mix_split
from verdict.metrics import MetricRegistry
from verdict.query import Counters, Segment

REGISTRY = MetricRegistry.load(Path(__file__).resolve().parents[1] / "config" / "metrics.yaml")
FILL_RATE = REGISTRY.metric("fill_rate")
REQUESTS = REGISTRY.metric("requests")


def cell(requests: float, fills: float) -> Counters:
    return Counters(requests=requests, fills=fills)


def partition(**segments: tuple[float, float]) -> dict[Segment, Counters]:
    """Keyword name -> (requests, fills), keyed by a one-dimensional segment."""
    return {
        Segment.of(os_version=name): cell(req, fills)
        for name, (req, fills) in segments.items()
    }


class TestTheIdentityHolds:
    def test_the_three_terms_reproduce_the_aggregate_movement(self):
        split = rate_mix_split(
            FILL_RATE,
            observed=partition(android=(6000, 4200), ios=(4000, 3400)),
            baseline=partition(android=(5000, 3750), ios=(5000, 4250)),
        )
        assert split is not None
        assert split.total == pytest.approx(split.after - split.before)

    def test_it_holds_when_a_segment_appears(self):
        split = rate_mix_split(
            FILL_RATE,
            observed=partition(android=(5000, 3750), web=(2000, 400)),
            baseline=partition(android=(5000, 3750)),
        )
        assert split is not None
        assert split.total == pytest.approx(split.after - split.before)

    def test_it_holds_when_a_segment_disappears(self):
        split = rate_mix_split(
            FILL_RATE,
            observed=partition(android=(5000, 3750)),
            baseline=partition(android=(5000, 3750), web=(2000, 400)),
        )
        assert split is not None
        assert split.total == pytest.approx(split.after - split.before)


class TestPureMixIsNotReportedAsDegradation:
    """The case the whole module exists for.

    Every segment fills exactly as well as it did before. Traffic moved toward the one that was
    always worse, so the aggregate falls. Nothing is broken and nobody should be paged.
    """

    @staticmethod
    def split():
        # Android fills at 0.80, iOS at 0.60, in both periods. Only the shares change.
        return rate_mix_split(
            FILL_RATE,
            baseline=partition(android=(8000, 6400), ios=(2000, 1200)),
            observed=partition(android=(2000, 1600), ios=(8000, 4800)),
        )

    def test_the_aggregate_really_does_fall(self):
        got = self.split()
        assert got.before == pytest.approx(0.76)
        assert got.after == pytest.approx(0.64)

    def test_no_segment_changed_its_own_rate(self):
        got = self.split()
        for seg in got.segments:
            assert seg.rate_before == pytest.approx(seg.rate_after)

    def test_the_entire_movement_is_attributed_to_mix(self):
        got = self.split()
        assert got.rate == pytest.approx(0.0)
        assert got.interaction == pytest.approx(0.0)
        assert got.mix == pytest.approx(got.after - got.before)
        assert got.dominant == "mix"
        assert got.mix_share == pytest.approx(1.0)

    def test_it_says_so_in_words(self):
        assert "Traffic mix, not rates" in self.split().describe()


class TestPureRateChangeIsReportedAsSuch:
    @staticmethod
    def split():
        # Shares identical; Android's own fill rate collapses from 0.80 to 0.40.
        return rate_mix_split(
            FILL_RATE,
            baseline=partition(android=(5000, 4000), ios=(5000, 3000)),
            observed=partition(android=(5000, 2000), ios=(5000, 3000)),
        )

    def test_the_movement_is_all_rate(self):
        got = self.split()
        assert got.mix == pytest.approx(0.0)
        assert got.interaction == pytest.approx(0.0)
        assert got.rate == pytest.approx(got.after - got.before)
        assert got.dominant == "rate"

    def test_the_guilty_segment_carries_it(self):
        got = self.split()
        worst = got.segments[0]
        assert worst.segment.as_dict()["os_version"] == "android"
        assert worst.rate == pytest.approx(-0.20)


class TestCancellationIsVisible:
    """A rate gain hidden by a mix loss. The net is nearly nothing; both forces are real."""

    def test_a_flat_aggregate_can_still_be_two_large_opposing_effects(self):
        got = rate_mix_split(
            FILL_RATE,
            baseline=partition(android=(5000, 4000), ios=(5000, 3000)),
            observed=partition(android=(3000, 2640), ios=(7000, 4340)),
        )
        assert abs(got.after - got.before) < 0.02
        # Neither term is small, which is the fact a net-only view destroys.
        assert abs(got.rate) > 0.03
        assert abs(got.mix) > 0.03
        # And the share is measured against everything that happened, not against the net,
        # so it stays a proportion rather than exploding.
        assert 0.0 <= got.mix_share <= 1.0


class TestItRefusesWhenItCannot:
    def test_a_count_metric_has_no_aggregate_rate_to_split(self):
        assert rate_mix_split(REQUESTS, observed=partition(a=(1, 1)),
                              baseline=partition(a=(1, 1))) is None

    def test_an_empty_period_yields_nothing(self):
        assert rate_mix_split(FILL_RATE, observed={}, baseline=partition(a=(10, 5))) is None
        assert rate_mix_split(FILL_RATE, observed=partition(a=(10, 5)), baseline={}) is None
