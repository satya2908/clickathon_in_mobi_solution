"""Tests for localization.

Built on a synthetic world with a known ground truth, so the tests can assert that the right
segment is accused *and* that the plausible wrong ones are rejected for the right reason.
Every scenario here mirrors a failure mode present in the real dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from verdict.config import DetectionConfig, LocalizationConfig
from verdict.detect import Finding
from verdict.localize import (
    Candidate,
    HistoryCache,
    Localizer,
    exonerate,
    expected_counters,
    maximality_check,
    minimality_check,
    sufficiency_check,
    trim_mask,
)
from verdict.metrics import MetricRegistry
from verdict.query import Counters, Segment, Window, subtract
from verdict.schema import TOTAL_COMBO
from verdict.stats import TestResult

METRICS_PATH = Path(__file__).resolve().parents[1] / "config" / "metrics.yaml"
REGISTRY = MetricRegistry.load(METRICS_PATH)
FILL_RATE = REGISTRY.metric("fill_rate")
CTR = REGISTRY.metric("ctr")

WINDOW = Window(datetime(2026, 6, 23), datetime(2026, 6, 26), "1d")

BASE_RATE = 0.785
BAD_RATE = 0.55


def counters(requests: int, rate: float) -> Counters:
    """A cell with a given request volume and fill rate."""
    return Counters(requests=requests, fills=int(round(requests * rate)))


def weeks(observed: Counters, baseline: Counters, n: int = 4) -> list[Counters]:
    """One observation followed by n identical historical weeks."""
    return [observed, *[baseline] * n]


class FakeReader:
    """Serves pre-built lattice cells, so localization can be tested without a database."""

    def __init__(self, combos: dict[str, dict[Segment, list[Counters]]]) -> None:
        self.combos = combos

    def slice_with_history(self, combo, window, weeks_back):  # noqa: ARG002
        return self.combos.get(combo, {})

    def segment(self, segment, window):  # noqa: ARG002
        for cells in self.combos.values():
            if segment in cells:
                return cells[segment][0]
        return Counters()


# Requests in each (device_model, os_version) cell. Three values per dimension, not two,
# because maximality asks "did this candidate's siblings move with it?" and with a single
# sibling that question has only the answers 0% and 100%. Every dimension in the real dataset
# carries at least three values (publisher_tier and campaign_type are the smallest, at three),
# so a two-value world tests a shape that cannot occur and leaves the check permanently
# unevaluated.
GRID = {
    ("Galaxy S23", "Android 15"): 9_000,
    ("Galaxy S23", "iOS 18.1"): 3_000,
    ("Galaxy S23", "Android 14"): 3_000,
    ("Pixel 8", "Android 15"): 7_000,
    ("Pixel 8", "iOS 18.1"): 20_000,
    ("Pixel 8", "Android 14"): 8_000,
    ("Redmi 12", "Android 15"): 4_000,
    ("Redmi 12", "iOS 18.1"): 27_000,
    ("Redmi 12", "Android 14"): 19_000,
}

# The planted incident: Android 15 alone collapses, on every device.
AFFECTED_OS = "Android 15"


def build_world() -> dict[str, dict[Segment, list[Counters]]]:
    """Android 15 fill rate collapses. Galaxy S23 skews Android 15 and drops with it.

    Total traffic is 100,000. Android 15 is 20,000 of it and falls from 0.785 to 0.55. Galaxy
    S23 is 15,000 requests, 9,000 of them Android 15, so it drops to 0.644 without being the
    cause. That is the exact shape that defeats ranking by effect size: the passenger's move is
    large, significant, and entirely explained by someone else.

    Marginals are summed from GRID rather than written out, so the one-way cells, the two-way
    cells and the grand total cannot silently disagree. Hand-written totals that drift a few
    hundred requests from their parts would make the sufficiency arithmetic wrong in a way that
    looks like a threshold problem.
    """

    def cell(requests: int, os_version: str) -> tuple[Counters, Counters]:
        rate = BAD_RATE if os_version == AFFECTED_OS else BASE_RATE
        return counters(requests, rate), counters(requests, BASE_RATE)

    two_way: dict[Segment, list[Counters]] = {}
    by_device: dict[str, list[Counters]] = {}
    by_os: dict[str, list[Counters]] = {}
    total_obs = total_base = Counters()

    for (device, os_version), requests in GRID.items():
        obs, base = cell(requests, os_version)
        two_way[Segment.of(device_model=device, os_version=os_version)] = weeks(obs, base)

        d = by_device.setdefault(device, [Counters(), Counters()])
        d[0], d[1] = d[0] + obs, d[1] + base
        o = by_os.setdefault(os_version, [Counters(), Counters()])
        o[0], o[1] = o[0] + obs, o[1] + base
        total_obs, total_base = total_obs + obs, total_base + base

    return {
        "__all__": {Segment.total(): weeks(total_obs, total_base)},
        "os_version": {
            Segment.of(os_version=name): weeks(obs, base) for name, (obs, base) in by_os.items()
        },
        "device_model": {
            Segment.of(device_model=name): weeks(obs, base)
            for name, (obs, base) in by_device.items()
        },
        "device_model|os_version": two_way,
    }


class TestWorldIsSelfConsistent:
    """If the fixture's marginals disagree with its cells, every assertion below is meaningless.

    These are not tests of the system; they are tests of the test, and they exist because the
    previous version of this fixture wrote its totals by hand.
    """

    def test_marginals_sum_to_the_grand_total(self):
        cells = build_world()
        total = cells["__all__"][Segment.total()][0]
        for combo in ("os_version", "device_model"):
            summed = Counters()
            for history in cells[combo].values():
                summed = summed + history[0]
            assert summed == total, f"{combo} marginal does not reconstruct the total"

    def test_every_dimension_has_enough_values_to_test_maximality(self):
        cells = build_world()
        for combo in ("os_version", "device_model"):
            assert len(cells[combo]) >= 3

    def test_the_planted_incident_is_where_the_tests_assume(self):
        cells = build_world()
        a15 = cells["os_version"][Segment.of(os_version=AFFECTED_OS)]
        assert a15[0].value(FILL_RATE) == pytest.approx(BAD_RATE, abs=0.001)
        assert a15[1].value(FILL_RATE) == pytest.approx(BASE_RATE, abs=0.001)


def make_candidate(cells, combo, segment):
    from verdict.localize import Candidate

    history = cells[combo][segment]
    observed, expected = history[0], expected_counters(history[1:])
    return Candidate(
        segment=segment,
        observed=observed,
        expected=expected,
        observed_value=observed.value(FILL_RATE),
        expected_value=expected.value(FILL_RATE),
    )


class TestExpectedCounters:
    def test_identical_weeks_reproduce_themselves(self):
        baseline = counters(20_000, BASE_RATE)
        assert expected_counters([baseline] * 4) == baseline

    def test_every_counter_is_averaged_over_the_same_weeks(self):
        """Taking requests from one set of weeks and fills from another would produce an
        expected fill rate that no week actually had."""
        history = [
            counters(1_000, 0.80),
            counters(1_000, 0.79),
            counters(1_000, 0.81),
            counters(1_000, 0.20),  # contaminated
        ]
        result = expected_counters(history, trim_mask(history, FILL_RATE))
        assert result.value(FILL_RATE) == pytest.approx(0.80, abs=0.01)

    def test_empty_history_is_empty(self):
        assert expected_counters([]) == Counters()


class TestTrimMask:
    def test_drops_the_single_week_furthest_from_the_median(self):
        history = [counters(1_000, r) for r in (0.80, 0.79, 0.81, 0.20)]
        assert trim_mask(history, FILL_RATE) == (0, 1, 2)

    def test_keeps_everything_when_no_week_is_contaminated(self):
        history = [counters(1_000, r) for r in (0.80, 0.79, 0.81, 0.80)]
        assert len(trim_mask(history, FILL_RATE)) == 3  # one is always dropped when trimming

    def test_refuses_to_trim_below_three_weeks(self):
        """Dropping one of two leaves a single sample, which is not a baseline."""
        history = [counters(1_000, 0.80), counters(1_000, 0.20)]
        assert trim_mask(history, FILL_RATE) == (0, 1)

    def test_excludes_weeks_with_no_traffic_in_the_metric_denominator(self):
        history = [counters(1_000, 0.80), Counters(), counters(1_000, 0.79), counters(1_000, 0.81)]
        assert 1 not in trim_mask(history, FILL_RATE)

    def test_trims_on_the_metric_under_investigation_not_fill_rate(self):
        """The defect this replaced hardcoded fills/requests regardless of the metric, so a CTR
        case discarded whichever week was odd for a quantity nobody had asked about -- while
        leaving in the week that actually distorted the CTR baseline.

        Week 1 is the CTR outlier and week 3 the fill-rate outlier. Trimming for CTR must drop
        week 1, which a fill-rate criterion would have kept.
        """
        history = [
            Counters(requests=1_000, fills=800, impressions=800, clicks=80),
            Counters(requests=1_000, fills=800, impressions=800, clicks=8),  # CTR outlier
            Counters(requests=1_000, fills=800, impressions=800, clicks=80),
            Counters(requests=1_000, fills=200, impressions=200, clicks=20),  # fill-rate outlier
        ]
        assert trim_mask(history, CTR) == (0, 2, 3)
        assert trim_mask(history, FILL_RATE) == (0, 1, 2)


class TestCounterfactualsStayNested:
    """The defect that motivated the shared mask.

    Trim was chosen per segment, so a parent could drop week two while a candidate dropped week
    four. `parent_expected - candidate_expected` then subtracted quantities averaged over
    different weeks, nothing forced the candidate's requests below the parent's, and the old
    per-field clamp turned the negative result into a rate outside [0, 1] -- a confident,
    precise, meaningless verdict with no error raised anywhere.
    """

    def _world(self):
        """Parent = child + sibling, built additively so containment holds every week.

        The child has a bad week 3; the sibling has a bad week 1. Because the sibling is the
        larger of the two, the parent's own worst week is week 1 -- so the parent and the child
        disagree about which week to discard, which is the whole setup.

        The sibling's fill rate over weeks 0, 2 and 3 is exactly 0.80, and that is the number a
        correct counterfactual has to recover when the child is removed from the parent.
        """
        child = [counters(100, r) for r in (0.80, 0.80, 0.80, 0.20)]
        sibling = [counters(900, r) for r in (0.80, 0.20, 0.80, 0.80)]
        parent = [c + s for c, s in zip(child, sibling, strict=True)]
        return parent, child

    def test_the_two_masks_really_do_diverge(self):
        parent, child = self._world()
        assert trim_mask(parent, FILL_RATE) == (0, 2, 3)
        assert trim_mask(child, FILL_RATE) == (0, 1, 2)

    def test_a_shared_mask_recovers_the_sibling_rate_exactly(self):
        parent, child = self._world()
        mask = trim_mask(parent, FILL_RATE)
        remainder = subtract(expected_counters(parent, mask), expected_counters(child, mask))
        assert remainder is not None
        assert remainder.value(FILL_RATE) == pytest.approx(0.80, abs=1e-9)

    def test_per_segment_masks_get_the_wrong_answer(self):
        """The defect, as a number rather than an exception.

        Nothing raises and nothing looks odd -- the counterfactual is simply 2.2 points off,
        because the parent's expectation excludes week 1 while the child's includes it. A
        sufficiency score built on this is precise and wrong, which is the failure mode that
        matters most for something calling itself an autonomous investigator.
        """
        parent, child = self._world()
        p = expected_counters(parent, trim_mask(parent, FILL_RATE))
        c = expected_counters(child, trim_mask(child, FILL_RATE))
        remainder = subtract(p, c)
        assert remainder is not None
        assert remainder.value(FILL_RATE) == pytest.approx(0.7778, abs=1e-4)

    def test_subtraction_refuses_rather_than_manufacturing_an_impossible_rate(self):
        """The reviewer's example. (1000 req, 800 fills) - (900 req, 100 fills) is (100, 700),
        in which no field is negative and the fill rate is 700%. Non-negativity is not the
        invariant; funnel coherence is."""
        minuend = Counters(requests=1_000, fills=800)
        subtrahend = Counters(requests=900, fills=100)
        naive = Counters(requests=100, fills=700)
        assert naive.value(FILL_RATE) == pytest.approx(7.0)  # what per-field clamping produced

        assert subtract(minuend, subtrahend) is None
        assert (minuend - subtrahend) == Counters()
        assert (minuend - subtrahend).value(FILL_RATE) is None

    def test_coherence_follows_the_funnel_not_just_the_sign(self):
        assert Counters(requests=100, fills=80, impressions=60, clicks=6).coherent
        assert not Counters(requests=100, fills=700).coherent
        assert not Counters(requests=100, fills=80, impressions=90).coherent
        assert not Counters(requests=100, fills=80, impressions=60, clicks=61).coherent
        assert not Counters(requests=-1).coherent
        assert not Counters(requests=100, fills=80, revenue=-0.01).coherent

    def test_a_genuine_removal_still_subtracts_exactly(self):
        minuend = Counters(requests=1_000, fills=800, impressions=800, clicks=80)
        subtrahend = Counters(requests=400, fills=320, impressions=320, clicks=32)
        out = subtract(minuend, subtrahend)
        assert out == Counters(requests=600, fills=480, impressions=480, clicks=48)
        assert out.value(FILL_RATE) == pytest.approx(0.80)


class TestSufficiency:
    def test_the_cause_explains_the_movement_and_the_passenger_does_not(self):
        """The single measurement this whole approach rests on.

        Removing Android 15 restores the global fill rate exactly. Removing Galaxy S23 -- which
        dropped visibly and significantly -- leaves most of the movement in place. Ranking by
        effect size cannot distinguish these two; the counterfactual separates them cleanly.
        """
        cells = build_world()
        total = cells["__all__"][Segment.total()]
        parent_obs, parent_exp = total[0], expected_counters(total[1:])

        cause = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))
        passenger = make_candidate(cells, "device_model", Segment.of(device_model="Galaxy S23"))

        cause_check = sufficiency_check(FILL_RATE, parent_obs, parent_exp, cause, 0.60)
        passenger_check = sufficiency_check(FILL_RATE, parent_obs, parent_exp, passenger, 0.60)

        assert cause_check.state == "pass"
        assert cause_check.score == pytest.approx(1.0, abs=0.01)

        assert passenger_check.state == "fail"
        assert passenger_check.score == pytest.approx(0.35, abs=0.05)

        # The passenger is not a subtle effect: it dropped 18%, which any ranking would rank
        # highly. Its innocence is only visible through the counterfactual.
        assert passenger.relative_effect < -0.15

    def test_a_flat_parent_makes_sufficiency_unknown_not_failed(self):
        """A compensating pair leaves the parent unmoved. There is nothing for a
        counterfactual to restore, and reporting that as a failed test would be wrong."""
        cells = build_world()
        flat = counters(100_000, BASE_RATE)
        candidate = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))
        check = sufficiency_check(FILL_RATE, flat, flat, candidate, 0.60)
        assert check.state == "unknown"
        assert "did not move" in check.detail

    def test_removing_everything_is_unknown_rather_than_perfect(self):
        cells = build_world()
        total = cells["__all__"][Segment.total()]
        parent_obs, parent_exp = total[0], expected_counters(total[1:])
        whole = make_candidate(cells, "__all__", Segment.total())
        assert sufficiency_check(FILL_RATE, parent_obs, parent_exp, whole, 0.60).state == "unknown"


class TestMinimality:
    def test_rejects_a_candidate_whose_child_carries_everything(self):
        """Android 15 looks sufficient, but if only Android 15 on Galaxy S23 had moved, then
        removing that child would flatten Android 15 entirely and the honest answer is the
        child."""
        a15_bad_only_on_s23 = Counters(
            requests=20_000, fills=int(round(9_000 * BAD_RATE + 11_000 * BASE_RATE))
        )
        cells = {
            "os_version": {
                Segment.of(os_version="Android 15"): weeks(
                    a15_bad_only_on_s23, counters(20_000, BASE_RATE)
                )
            },
            "device_model|os_version": {
                Segment.of(device_model="Galaxy S23", os_version="Android 15"): weeks(
                    counters(9_000, BAD_RATE), counters(9_000, BASE_RATE)
                ),
                Segment.of(device_model="Pixel 8", os_version="Android 15"): weeks(
                    counters(11_000, BASE_RATE), counters(11_000, BASE_RATE)
                ),
            },
        }
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        candidate = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))

        check = minimality_check(FILL_RATE, candidate, history, REGISTRY, 0.30)
        assert check.state == "fail"
        assert check.score == pytest.approx(0.0, abs=0.02)
        assert "too broad" in check.detail

    def test_accepts_a_candidate_whose_deviation_is_spread_across_its_children(self):
        cells = build_world()
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        candidate = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))
        check = minimality_check(FILL_RATE, candidate, history, REGISTRY, 0.30)
        assert check.state == "pass"

    def test_two_dimensional_candidates_are_untested_not_passed(self):
        """The lattice carries one and two-dimensional cells. Testing minimality on a
        two-dimensional candidate would need a three-dimensional child that does not exist,
        and claiming a pass would assert a check that never ran."""
        cells = build_world()
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        candidate = make_candidate(
            cells,
            "device_model|os_version",
            Segment.of(device_model="Galaxy S23", os_version="Android 15"),
        )
        check = minimality_check(FILL_RATE, candidate, history, REGISTRY, 0.30)
        assert check.state == "unknown"
        assert "three-dimensional" in check.detail

    def test_the_whole_population_always_fails(self):
        cells = build_world()
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        candidate = make_candidate(cells, "__all__", Segment.total())
        assert minimality_check(FILL_RATE, candidate, history, REGISTRY, 0.30).state == "fail"


class TestMaximality:
    def test_rejects_a_candidate_whose_siblings_all_moved_with_it(self):
        """Android 15 broke on every device. Naming one device model would send an operator to
        inspect hardware when the whole platform is affected."""
        cells = build_world()
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        candidate = make_candidate(
            cells,
            "device_model|os_version",
            Segment.of(device_model="Galaxy S23", os_version="Android 15"),
        )
        check = maximality_check(FILL_RATE, candidate, history, 0.50)
        assert check.state == "fail"
        assert "wider than this candidate" in check.detail

    def test_accepts_a_candidate_its_siblings_did_not_follow(self):
        cells = build_world()
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        candidate = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))
        check = maximality_check(FILL_RATE, candidate, history, 0.50)
        assert check.state == "pass"


class TestExonerationLedger:
    def test_predicts_what_an_innocent_bystander_should_read(self):
        """Clearing a segment becomes a falsifiable claim rather than an omission.

        Galaxy S23 is 60% Android 15. If Android 15 explains everything, S23 must read
        0.6 * 0.55 + 0.4 * 0.785 = 0.644 -- which is exactly what it does read.
        """
        cells = build_world()
        accused = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))
        bystander = make_candidate(cells, "device_model", Segment.of(device_model="Galaxy S23"))

        predicted, residual = exonerate(FILL_RATE, bystander, accused, overlap_share=0.6)
        assert predicted == pytest.approx(0.644, abs=0.001)
        assert abs(residual) < 0.005

    def test_refuses_to_clear_a_segment_carrying_a_second_cause(self):
        """The property that makes the ledger worth publishing.

        A segment with no Android 15 exposure that dropped anyway cannot be explained by the
        accused, and the residual says so instead of the segment quietly disappearing.
        """
        cells = build_world()
        accused = make_candidate(cells, "os_version", Segment.of(os_version="Android 15"))

        from verdict.localize import Candidate

        second_cause = Candidate(
            segment=Segment.of(region="APAC"),
            observed=counters(10_000, 0.60),
            expected=counters(10_000, BASE_RATE),
            observed_value=0.60,
            expected_value=BASE_RATE,
        )
        predicted, residual = exonerate(FILL_RATE, second_cause, accused, overlap_share=0.0)
        assert predicted == pytest.approx(BASE_RATE)
        assert abs(residual) > 0.005


class TestEndToEnd:
    def _localizer(self, cells):
        return Localizer(
            reader=FakeReader(cells),
            registry=REGISTRY,
            localization=LocalizationConfig(),
            detection=DetectionConfig(baseline_weeks=4),
        )

    def _finding(self, segment: Segment | None = None):
        segment = segment or Segment.total()
        return Finding(
            metric="fill_rate",
            segment=segment,
            window=WINDOW,
            detector="temporal",
            test=TestResult(-5.0, 1e-7, 0.738, 0.785, -0.047, -0.06, "two_proportion"),
            observed_counters=Counters(),
            baseline_counters=Counters(),
            phi=1.0,
        )

    def test_accuses_the_cause_and_clears_the_passenger(self):
        cells = build_world()
        localizer = self._localizer(cells)
        localizer.cfg.holdout_enabled = False

        result = localizer.localize(self._finding())

        assert result.accused is not None
        assert result.accused.segment == Segment.of(os_version="Android 15")
        assert result.accused.sufficiency > 0.95

        by_segment = {c.segment: c for c in result.candidates}
        passenger = by_segment[Segment.of(device_model="Galaxy S23")]
        assert passenger.status != "accused"
        assert passenger.exoneration_residual is not None
        assert abs(passenger.exoneration_residual) < 0.005

    def test_reports_no_verdict_rather_than_the_best_of_a_bad_set(self):
        """When nothing survives the tests, silence is the correct output. A system that
        always names something teaches operators to ignore it."""
        flat = counters(100_000, BASE_RATE)
        cells = {
            "__all__": {Segment.total(): weeks(flat, flat)},
            "os_version": {
                Segment.of(os_version="Android 15"): weeks(
                    counters(20_000, BASE_RATE), counters(20_000, BASE_RATE)
                )
            },
        }
        localizer = self._localizer(cells)
        localizer.cfg.holdout_enabled = False
        result = localizer.localize(self._finding())
        assert result.mode == "structural_only"
        assert result.accused is None or result.accused.sufficiency == 0.0


class TestTheCaseQuotesTheCellItAccuses:
    """A verdict has to carry evidence about its own claim.

    The detector reports whichever cell trips first; localization then reasons over the whole
    lattice and often names a different one. Quoting the entry cell's p-value produces a case
    that accuses one segment while reporting another's test -- on the real corpus, a finance
    eCPM verdict certified by a structural anomaly in EU interstitials. The localizer re-tests
    the cell it settled on so the case can speak for itself.
    """

    def _localizer(self, cells):
        return Localizer(
            reader=FakeReader(cells),
            registry=REGISTRY,
            localization=LocalizationConfig(),
            detection=DetectionConfig(baseline_weeks=4),
        )

    def _entry_finding(self):
        """Deliberately attributed to a cell that is not the culprit, as the detector's would be."""
        return Finding(
            metric="fill_rate",
            segment=Segment.of(device_model="Galaxy S23"),
            window=WINDOW,
            detector="structural",
            test=TestResult(-9.0, 1e-19, 0.70, 0.785, -0.085, -0.108, "two_proportion"),
            observed_counters=Counters(),
            baseline_counters=Counters(),
            phi=1.0,
        )

    def test_the_confirmatory_test_describes_the_accused_cell(self):
        localizer = self._localizer(build_world())
        localizer.cfg.holdout_enabled = False

        result = localizer.localize(self._entry_finding())

        assert result.accused is not None
        assert result.accused_finding is not None
        assert result.accused_finding.segment == result.accused.segment
        assert result.accused_finding.segment != self._entry_finding().segment

    def test_it_is_marked_as_selected_after_the_fact(self):
        """It must not be pooled back into the detector's corrected family."""
        localizer = self._localizer(build_world())
        localizer.cfg.holdout_enabled = False

        result = localizer.localize(self._entry_finding())

        assert result.accused_finding is not None
        assert result.accused_finding.detector == "confirmatory"
        assert result.accused_finding.survives_correction is False
        assert result.accused_finding.notes.get("selected_post_hoc") is True

    def test_it_actually_measured_something(self):
        localizer = self._localizer(build_world())
        localizer.cfg.holdout_enabled = False

        result = localizer.localize(self._entry_finding())

        assert result.accused_finding is not None
        assert result.accused_finding.test.testable

    def test_a_cell_with_no_rows_yields_no_finding_rather_than_an_empty_one(self):
        """The guard that stops a declined test being passed off as evidence of calm."""
        cells = build_world()
        localizer = self._localizer(cells)
        history = HistoryCache(FakeReader(cells), WINDOW, 4)
        history.adopt_mask(cells[TOTAL_COMBO][Segment.total()][1:], FILL_RATE, trim=False)

        absent = Candidate(
            segment=Segment.of(os_version="Android 99"),
            observed=counters(1_000, 0.5),
            expected=counters(1_000, BASE_RATE),
            observed_value=0.5,
            expected_value=BASE_RATE,
        )

        assert localizer._confirm(FILL_RATE, absent, history, WINDOW) is None
