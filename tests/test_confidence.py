"""Tests for the confidence score, and mostly for one property of it.

The weighted average is arithmetic and would be hard to get wrong. What is easy to get wrong,
and what every one of these tests is really about, is the third state. A check that could not
be run is not a check that failed, and a score that cannot tell those apart is miscalibrated in
a direction nobody notices: it reports low confidence on small segments, which reads as the
system being appropriately cautious rather than as the system being unable to look.

So the ordering `fail < unknown < pass` is pinned directly, the renormalised weights are
asserted as explicit arithmetic rather than through a helper that could share the bug, and the
degenerate end -- almost everything unknown -- is pinned too, because that is where a
renormalising score would otherwise hand back a confident number computed from one component.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from verdict.confidence import Confidence, score
from verdict.config import ConfidenceConfig
from verdict.detect import Finding
from verdict.localize import Candidate, Check, Localization
from verdict.query import Counters, Segment, Window
from verdict.stats import TestResult

WINDOW = Window(datetime(2026, 6, 23), datetime(2026, 6, 24), "1h")
CFG = ConfidenceConfig()

# A check spec is (state, score), or None to leave the key out of the dict entirely. Both
# absences must behave identically, and the tests below assert that they do.
CheckSpec = tuple[str, float | None] | None

UNKNOWN: CheckSpec = ("unknown", None)


def check(name: str, spec: CheckSpec) -> Check | None:
    if spec is None:
        return None
    state, value = spec
    if value is None:
        detail = f"The {name} test could not be run on this candidate."
    else:
        detail = f"The {name} test returned {value:.2f} and was recorded as a {state}."
    return Check(name, state, value, detail)


def counters(rate: float, requests: int = 10_000) -> Counters:
    return Counters(requests=requests, fills=int(round(requests * rate)))


def candidate(
    label: str,
    *,
    observed: float,
    expected: float = 0.78,
    status: str = "considered",
    **specs: CheckSpec,
) -> Candidate:
    """One lattice cell with whatever subset of checks the test needs."""
    checks = {}
    for name, spec in specs.items():
        built = check(name, spec)
        if built is not None:
            checks[name] = built
    return Candidate(
        segment=Segment.of(country=label),
        observed=counters(observed),
        expected=counters(expected),
        observed_value=observed,
        expected_value=expected,
        checks=checks,
        status=status,
    )


def finding(
    *,
    p_value: float = 1e-6,
    phi: float = 1.0,
    survives: bool = True,
    weeks_kept: int = 3,
    weeks_seen: int = 4,
    model: str = "two_proportion",
    screening: str = "benjamini_hochberg",
) -> Finding:
    return Finding(
        metric="fill_rate",
        segment=Segment.of(country="A"),
        window=WINDOW,
        detector="temporal",
        test=TestResult(
            z=-9.4,
            p_value=p_value,
            observed=0.55,
            expected=0.78,
            absolute_effect=-0.23,
            relative_effect=-0.2949,
            model=model,
        ),
        observed_counters=counters(0.55),
        baseline_counters=counters(0.78),
        phi=phi,
        weeks_kept=weeks_kept,
        weeks_seen=weeks_seen,
        survives_correction=survives,
        screening=screening,
    )


def localization(
    accused: Candidate | None,
    *others: Candidate,
    mode: str = "explain_away",
) -> Localization:
    candidates = [c for c in (accused, *others) if c is not None]
    return Localization(
        metric="fill_rate",
        window=WINDOW,
        parent=Segment.total(),
        parent_observed=0.60,
        parent_expected=0.78,
        parent_deviation=-0.18,
        accused=accused,
        candidates=candidates,
        mode=mode,
    )


def case(
    *,
    sufficiency: CheckSpec = ("pass", 0.95),
    minimality: CheckSpec = ("pass", 0.90),
    maximality: CheckSpec = ("pass", 0.90),
    holdout: CheckSpec = ("pass", 0.80),
    rival_sufficiency: float | None = 0.19,
    rival_status: str = "partial",
    mode: str = "explain_away",
    observed: float = 0.546,
    rival_observed: float = 0.7332,
) -> Localization:
    """The textbook shape: one accused cell and one runner-up that explains far less.

    The default rates are chosen so that the two scales separation can fall back between agree:
    the accused explains 0.95 of the parent against the rival's 0.19, and moves -30% against the
    rival's -6%. Both give a margin of 0.80, so a test that changes which scale is in use is
    testing the scale rather than a coincidence in the fixture.
    """
    accused = candidate(
        "A",
        observed=observed,
        status="accused",
        sufficiency=sufficiency,
        minimality=minimality,
        maximality=maximality,
        holdout=holdout,
    )
    rival = candidate(
        "B",
        observed=rival_observed,
        status=rival_status,
        sufficiency=None if rival_sufficiency is None else ("fail", rival_sufficiency),
    )
    return localization(accused, rival, mode=mode)


def component(confidence: Confidence, name: str):
    return next(c for c in confidence.components if c.name == name)


class TestTextbookCases:
    def test_a_clean_case_scores_high_and_publishes(self):
        """Every check passed, the p-value is six decades deep, the effect reproduced on the
        held-out half and the runner-up explains a fifth of what the accused does.

        0.25*1.00 + 0.30*0.95 + 0.15*0.90 + 0.15*0.80 + 0.15*0.80 = 0.91
        """
        conf = score(case(), finding(), CFG)
        assert conf.value == 0.91
        assert conf.publishable
        assert conf.components_scored == 5
        assert conf.components_total == 5
        assert conf.caveat == ""

    def test_a_weak_case_scores_low_and_does_not_publish(self):
        """The mirror image: a p-value at the uncorrected threshold on a cell ten times
        overdispersed, a segment explaining a fifth of the parent's movement, an effect that
        reversed direction across the window, and a runner-up almost as good.

        0.25*0.10 + 0.30*0.20 + 0.15*0.40 + 0.15*0.00 + 0.15*0.10 = 0.16
        """
        weak = case(
            sufficiency=("fail", 0.20),
            minimality=("pass", 0.40),
            maximality=("pass", 0.50),
            holdout=("fail", 0.0),
            rival_sufficiency=0.18,
            observed=0.70,
        )
        conf = score(weak, finding(p_value=0.01, phi=10.0), CFG)
        assert conf.value == 0.16
        assert not conf.publishable
        assert conf.components_scored == 5

    def test_every_component_is_reported_even_when_it_scored_zero(self):
        """A component dropped from the list would be indistinguishable from one that failed,
        and the failing component is the one an operator most needs to see."""
        conf = score(case(holdout=("fail", 0.0)), finding(), CFG)
        stability = component(conf, "stability")
        assert stability.state == "scored"
        assert stability.score == 0.0
        assert stability.weight == 0.15


class TestUnknownIsNeitherFailureNorSuccess:
    """The ordering these tests pin is the whole design.

    All three cases below are the same candidate with the same evidence everywhere else. Only
    the minimality check differs: it passed outright, it could not be run, or it failed.
    """

    PASSED = 0.925  # 0.25*1.00 + 0.30*0.95 + 0.15*1.00 + 0.15*0.80 + 0.15*0.80
    UNKNOWN = 0.911765  # (0.25*1.00 + 0.30*0.95 + 0.15*0.80 + 0.15*0.80) / 0.85
    FAILED = 0.775  # 0.25*1.00 + 0.30*0.95 + 0.15*0.00 + 0.15*0.80 + 0.15*0.80

    def test_the_ordering_holds(self):
        """An untestable candidate must land above one that was tested and failed, and below one
        that was tested and passed. Anything else either punishes a segment for being too small
        to test -- which is the segment an operator most needs told about -- or rewards it for
        being unmeasurable, which would make the thinnest cells in the lattice look like the
        best-evidenced ones.
        """
        passed = score(case(minimality=("pass", 1.0)), finding(), CFG).value
        unknown = score(case(minimality=UNKNOWN), finding(), CFG).value
        failed = score(case(minimality=("fail", 0.0)), finding(), CFG).value

        assert failed < unknown < passed
        assert (passed, unknown, failed) == (self.PASSED, self.UNKNOWN, self.FAILED)

    def test_an_unknown_is_not_scored_as_zero(self):
        """The failing case and the unknown case share a numerator: 0.775. The only difference
        between them is that the unknown withdrew its weight and the rest were renormalised over
        0.85. That is the entire mechanism, stated as arithmetic."""
        unknown = score(case(minimality=UNKNOWN), finding(), CFG)
        assert unknown.value == pytest.approx(0.775 / 0.85)
        assert unknown.value > self.FAILED

    def test_an_unknown_is_not_scored_as_one(self):
        """Renormalising leaves the score where the remaining evidence puts it. Since the rest
        of this case averages below 1.0, the unknown cannot lift it to the top of the range."""
        unknown = score(case(minimality=UNKNOWN), finding(), CFG)
        assert unknown.value < 1.0
        assert unknown.value < self.PASSED

    def test_an_unknown_reduces_the_count_of_scored_components(self):
        conf = score(case(minimality=UNKNOWN), finding(), CFG)
        assert conf.components_scored == 4
        assert conf.components_total == 5
        assert component(conf, "minimality").state == "unknown"
        assert component(conf, "sufficiency").state == "scored"

    def test_the_remaining_weights_are_renormalised_to_sum_to_one(self):
        conf = score(case(minimality=UNKNOWN), finding(), CFG)
        weights = {c.name: c.weight for c in conf.components}

        assert weights["minimality"] == 0.0
        assert weights["significance"] == pytest.approx(0.25 / 0.85)
        assert weights["sufficiency"] == pytest.approx(0.30 / 0.85)
        assert weights["stability"] == pytest.approx(0.15 / 0.85)
        assert weights["separation"] == pytest.approx(0.15 / 0.85)
        assert sum(weights.values()) == pytest.approx(1.0)

        # Renormalisation must not reorder the components against each other. Sufficiency is
        # twice stability before it and must stay twice stability after it.
        assert weights["sufficiency"] == pytest.approx(2.0 * weights["stability"])

    def test_an_unknown_surfaces_a_caveat_naming_it(self):
        """A confidence of 0.91 from five components and one from four are different claims, so
        the difference has to be legible without reading the component list."""
        conf = score(case(minimality=UNKNOWN), finding(), CFG)
        assert "minimality" in conf.caveat
        assert "1 of 5 components could not be evaluated" in conf.caveat
        assert "85%" in conf.caveat

    def test_a_missing_check_is_the_same_as_an_explicit_unknown(self):
        """The localizer omits a check it never reached, and records an explicit unknown for one
        it reached and could not run. Neither is evidence of anything, so neither may score."""
        absent = score(case(minimality=None), finding(), CFG)
        explicit = score(case(minimality=UNKNOWN), finding(), CFG)
        assert absent.value == explicit.value == self.UNKNOWN
        assert absent.components_scored == explicit.components_scored == 4


class TestDegenerateCoverage:
    def test_two_unknowns_cap_the_score_below_what_the_rest_would_give(self):
        """Renormalising alone would report 0.49/0.55 = 0.891 here from three components, which
        claims a rounded picture the evidence does not support. 55% of the weight was scored, so
        the ceiling is 0.55/0.70 and the score is held to it.
        """
        conf = score(case(sufficiency=UNKNOWN, minimality=UNKNOWN), finding(), CFG)
        assert conf.components_scored == 3
        assert conf.value == pytest.approx(0.55 / 0.70)
        assert conf.value == 0.785714
        assert "Capped at 0.7857" in conf.caveat
        assert "uncapped 0.8909" in conf.caveat

    def test_one_missing_component_is_not_capped(self):
        """The ceiling starts biting at two holes, not one. With the shipped weights, losing the
        single heaviest component still leaves 70% of the weight and four components bounding
        the answer from four directions, which is a picture worth reporting in full."""
        conf = score(case(sufficiency=UNKNOWN), finding(), CFG)
        assert conf.components_scored == 4
        assert "Capped" not in conf.caveat
        assert conf.value == pytest.approx((0.25 * 1.0 + 0.15 * 0.9 + 0.15 * 0.8 * 2) / 0.70)
        assert conf.value == 0.892857

    def test_a_single_component_cannot_publish_however_good_it_is(self):
        """A perfect p-value and nothing else. Renormalisation alone would return 1.0, which
        would be the highest confidence the system can express, awarded to the candidate about
        which the least is known."""
        alone = localization(candidate("A", observed=0.55, status="accused"))
        conf = score(alone, finding(p_value=1e-30), CFG)

        assert conf.components_scored == 1
        assert component(conf, "significance").score == 1.0
        assert conf.value == pytest.approx(0.25 / 0.70)
        assert not conf.publishable
        assert "fewer than 3 components were scored" in conf.caveat

    def test_two_components_are_still_short_of_the_publication_floor(self):
        """The five components are five different ways to be wrong, not five readings of one
        quantity, so a majority of them going unchecked is disqualifying on its own."""
        thin = case(sufficiency=UNKNOWN, minimality=UNKNOWN, holdout=UNKNOWN)
        conf = score(thin, finding(), CFG)
        assert conf.components_scored == 2
        assert not conf.publishable

    def test_nothing_scored_is_an_absence_rather_than_a_zero(self):
        conf = score(localization(None), finding(), CFG)
        assert conf.value == 0.0
        assert conf.components_scored == 0
        assert not conf.publishable
        assert all(c.state == "unknown" and c.weight == 0.0 for c in conf.components)
        assert "absence of a score rather than a low one" in conf.caveat

    def test_the_cap_never_raises_a_weak_score(self):
        """The ceiling is one-sided on purpose. A candidate that scored badly on the components
        it could be measured on must not be lifted by the ones it could not."""
        bad = case(
            sufficiency=("fail", 0.05),
            minimality=UNKNOWN,
            holdout=("fail", 0.0),
            rival_sufficiency=0.05,
        )
        conf = score(bad, finding(p_value=0.05), CFG)
        assert conf.value < 0.1
        assert not conf.publishable


class TestSignificance:
    def test_evidence_is_counted_in_decades(self):
        """p is not linear in what it tells you. Anchored at zero credit for p = 0.1 and full
        credit for p = 1e-6, a finding that scraped past the correction and one that cleared it
        by orders of magnitude cannot land on the same number."""
        scraped = component(score(case(), finding(p_value=0.01), CFG), "significance").score
        overwhelming = component(score(case(), finding(p_value=1e-6), CFG), "significance").score
        assert scraped == pytest.approx(0.2)
        assert overwhelming == 1.0
        assert component(score(case(), finding(p_value=0.1), CFG), "significance").score == 0.0

    def test_an_underflowed_p_value_is_not_treated_as_missing_evidence(self):
        """normal_sf returns 0.0 past about z = 39 rather than losing precision. Taking a
        logarithm of it without a floor would raise, and treating it as p = 1 would report the
        strongest result in the corpus as the weakest."""
        assert component(score(case(), finding(p_value=0.0), CFG), "significance").score == 1.0

    def test_overdispersion_discounts_the_evidence(self):
        """The p-value has already been widened by phi. This is the second-order penalty: phi
        itself is estimated from a handful of weekly samples, so a cell needing a tenfold
        inflation is one whose own history is doing the arguing."""
        clean = component(score(case(), finding(p_value=1e-6, phi=1.0), CFG), "significance")
        noisy = component(score(case(), finding(p_value=1e-6, phi=10.0), CFG), "significance")
        assert clean.score == 1.0
        assert noisy.score == pytest.approx(0.5)
        assert "Dispersion measured at 10.00x" in noisy.detail

    def test_a_finding_that_failed_correction_is_capped(self):
        rejected = component(
            score(case(), finding(p_value=1e-30, survives=False), CFG), "significance"
        )
        assert rejected.score == 0.25
        assert "did not survive the false-discovery correction" in rejected.detail

    def test_a_structural_finding_is_capped_without_claiming_it_failed_a_correction(self):
        """Structural findings never enter the Benjamini-Hochberg family at all.

        Saying they "did not survive" it describes a contest that never happened, which is the
        same class of false statement as claiming they survived it -- and the latter is what
        the default value used to produce.
        """
        got = component(
            score(
                case(),
                finding(p_value=1e-30, survives=False, screening="structural_z"),
                CFG,
            ),
            "significance",
        )
        assert got.score == 0.25
        assert "fixed structural threshold" in got.detail
        assert "did not survive" not in got.detail

    def test_a_post_hoc_confirmation_names_the_selection_effect(self):
        got = component(
            score(
                case(),
                finding(p_value=1e-30, survives=False, screening="post_hoc"),
                CFG,
            ),
            "significance",
        )
        assert got.score == 0.25
        assert "selection effect" in got.detail
        assert "did not survive" not in got.detail

    def test_a_test_that_never_ran_is_unknown_rather_than_insignificant(self):
        """log_ratio_test returns p = 1.0 and z = 0.0 when the segment's history was too short
        or too flat to test against. Scoring that as a p-value of one would report an untested
        segment as a tested and unremarkable one."""
        conf = score(case(), finding(model="log_ratio_insufficient"), CFG)
        assert component(conf, "significance").state == "unknown"
        assert conf.components_scored == 4


class TestStability:
    def test_a_holdout_that_was_never_run_is_unknown(self):
        """Holdout testing can be switched off, and a window can be too short to split. Neither
        is a failure to reproduce."""
        conf = score(case(holdout=None), finding(), CFG)
        assert component(conf, "stability").state == "unknown"
        assert conf.components_scored == 4

    def test_a_holdout_that_reversed_direction_scores_zero(self):
        """Here the check did run and the two halves disagreed, which is what an artefact of
        window choice looks like. That is a measurement, so it scores."""
        conf = score(case(holdout=("fail", 0.0)), finding(), CFG)
        stability = component(conf, "stability")
        assert stability.state == "scored"
        assert stability.score == 0.0

    def test_a_thin_baseline_discounts_the_reproduction(self):
        """Both halves are compared against expectations built from the same weekly history, so
        a thin baseline degrades the reproduction rather than being separate evidence."""
        full = component(score(case(), finding(weeks_kept=3, weeks_seen=4), CFG), "stability")
        thin = component(score(case(), finding(weeks_kept=1, weeks_seen=4), CFG), "stability")
        assert full.score == pytest.approx(0.80)
        assert thin.score == pytest.approx(0.80 * (0.5 + 0.5 / 3.0))
        assert "kept 1 of 4 historical weeks" in thin.detail

    def test_a_thin_baseline_cannot_erase_a_reproduction(self):
        """An effect that reproduced on a fresh half of the window is still evidence when the
        baseline is one week. It is weaker evidence, not absent evidence."""
        worst = component(score(case(), finding(weeks_kept=0, weeks_seen=1), CFG), "stability")
        assert worst.score == pytest.approx(0.80 * 0.5)

    def test_a_detector_that_keeps_no_weekly_baseline_is_not_penalised(self):
        """The structural detector fits a median polish across the grid instead of pooling
        weeks, so it leaves weeks_kept and weeks_seen at zero. Charging it for a field its
        detector never populates would score an absence as a defect."""
        conf = score(case(), finding(weeks_kept=0, weeks_seen=0), CFG)
        assert component(conf, "stability").score == pytest.approx(0.80)


class TestSeparation:
    def test_a_close_runner_up_collapses_the_margin(self):
        """Every other component grades the accused in isolation, and all of them can pass while
        a second candidate passes just as well. This is the only component that can see it."""
        clear = component(score(case(rival_sufficiency=0.19), finding(), CFG), "separation")
        contested = component(score(case(rival_sufficiency=0.93), finding(), CFG), "separation")
        assert clear.score == pytest.approx(0.80)
        assert contested.score == pytest.approx((0.95 - 0.93) / 0.95)
        assert contested.score < 0.03

    def test_the_margin_is_relative_rather_than_absolute(self):
        """A rival at 0.80 against an accused at 0.90 is a much closer call than one at 0.10
        against 0.20, though both are 0.10 apart."""
        near = component(
            score(case(sufficiency=("pass", 0.90), rival_sufficiency=0.80), finding(), CFG),
            "separation",
        )
        far = component(
            score(case(sufficiency=("pass", 0.20), rival_sufficiency=0.10), finding(), CFG),
            "separation",
        )
        assert near.score == pytest.approx(1.0 / 9.0)
        assert far.score == pytest.approx(0.5)

    def test_a_field_of_one_is_unknown_rather_than_a_clear_win(self):
        """Nothing was compared against, so there is no margin. Reporting 1.0 would award the
        highest possible separation to the candidate with the least competition examined."""
        alone = localization(
            candidate("A", observed=0.55, status="accused", sufficiency=("pass", 0.95))
        )
        conf = score(alone, finding(), CFG)
        assert component(conf, "separation").state == "unknown"

    def test_candidates_removed_by_a_breadth_check_are_not_near_misses(self):
        """A one-way parent that is guilty only through its child explains just as much of the
        movement as the child does, and minimality is what tells them apart. Counting it as a
        rival would report the lowest separation for the shape the breadth tests handle best.
        """
        accused = candidate(
            "A", observed=0.55, status="accused", sufficiency=("pass", 0.95)
        )
        parent = candidate(
            "B", observed=0.60, status="too_broad", sufficiency=("pass", 0.95)
        )
        conf = score(localization(accused, parent), finding(), CFG)
        separation = component(conf, "separation")
        assert separation.state == "scored"
        assert separation.score == 1.0
        assert "excluded by a breadth check or exonerated" in separation.detail

    def test_an_exonerated_candidate_is_not_a_near_miss_either(self):
        """The ledger predicted its movement from the accused's and was right. Penalising the
        accusation for a successful clearance inverts the evidence."""
        accused = candidate("A", observed=0.55, status="accused", sufficiency=("pass", 0.95))
        cleared = candidate("B", observed=0.74, status="cleared", sufficiency=("pass", 0.94))
        conf = score(localization(accused, cleared), finding(), CFG)
        assert component(conf, "separation").score == 1.0

    def test_a_rival_that_could_not_be_measured_does_not_count_as_explaining_nothing(self):
        """Candidate.sufficiency reports 0.0 both for a rival that explains nothing and for one
        that was never scored. Reading the second as the first would let the accused claim a
        margin over a candidate nobody measured."""
        conf = score(case(rival_sufficiency=None), finding(), CFG)
        separation = component(conf, "separation")
        assert separation.state == "unknown"
        assert "none could be placed on the same" in separation.detail

    def test_ties_are_broken_deterministically(self):
        """Two rivals reaching the same number must name the same one every time, so that two
        runs of the same investigation produce case files that diff cleanly."""
        accused = candidate("A", observed=0.55, status="accused", sufficiency=("pass", 0.95))
        first = candidate("Y", observed=0.74, status="partial", sufficiency=("fail", 0.40))
        second = candidate("X", observed=0.74, status="partial", sufficiency=("fail", 0.40))

        forwards = score(localization(accused, first, second), finding(), CFG)
        backwards = score(localization(accused, second, first), finding(), CFG)
        assert forwards.value == backwards.value
        assert "country=X" in component(forwards, "separation").detail
        assert "country=X" in component(backwards, "separation").detail


class TestStructuralOnlyMode:
    def test_a_parent_that_did_not_move_still_produces_a_score(self):
        """Two segments moving in opposite directions cancel at the total, so there is no parent
        deviation for a counterfactual to restore. Sufficiency is meaningless rather than
        failed, and the candidate is judged on the four components that still apply.

        (0.25*1.00 + 0.15*0.90 + 0.15*0.80 + 0.15*0.80) / 0.70 = 0.892857
        """
        structural = case(rival_sufficiency=None, mode="structural_only")
        conf = score(structural, finding(), CFG)

        assert conf.value == 0.892857
        assert conf.publishable
        assert conf.components_scored == 4
        assert component(conf, "sufficiency").state == "unknown"
        assert "no parent deviation" in component(conf, "sufficiency").detail

    def test_separation_falls_back_to_deviation_when_sufficiency_is_meaningless(self):
        """Ranking rivals by how much of the parent's movement they explain is nonsense when the
        parent did not move. The comparison moves to relative deviation, and the detail says so
        rather than leaving a reader to assume the usual scale."""
        accused = candidate(
            "A",
            observed=0.546,
            status="accused",
            sufficiency=UNKNOWN,
            minimality=("pass", 0.90),
            holdout=("pass", 0.80),
        )
        rival = candidate("B", observed=0.7332, status="partial", sufficiency=UNKNOWN)
        conf = score(localization(accused, rival, mode="structural_only"), finding(), CFG)

        separation = component(conf, "separation")
        assert separation.state == "scored"
        assert "relative deviation" in separation.detail
        # Accused moved -30% against the rival's -6%.
        assert separation.score == pytest.approx(0.8)

    def test_an_explicitly_unknown_sufficiency_check_is_honoured_in_either_mode(self):
        """The localizer records sufficiency as unknown when the parent did not move, so the two
        signals agree in practice. They must not disagree when a caller sets only one of them.
        """
        by_mode = score(case(mode="structural_only"), finding(), CFG)
        by_check = score(case(sufficiency=UNKNOWN, rival_sufficiency=None), finding(), CFG)
        assert component(by_mode, "sufficiency").state == "unknown"
        assert component(by_check, "sufficiency").state == "unknown"


class TestDeterminism:
    def test_the_same_inputs_produce_the_same_float(self):
        """A confidence written into a case file, compared against a threshold and read by an
        operator has to be one number in all three places."""
        first = score(case(), finding(), CFG)
        second = score(case(), finding(), CFG)
        assert first.value == second.value == 0.91
        assert [c.score for c in first.components] == [c.score for c in second.components]
        assert [c.weight for c in first.components] == [c.weight for c in second.components]
        assert first.explain() == second.explain()

    def test_components_are_always_reported_in_the_same_order(self):
        conf = score(case(minimality=UNKNOWN), finding(), CFG)
        assert [c.name for c in conf.components] == [
            "significance",
            "sufficiency",
            "minimality",
            "stability",
            "separation",
        ]

    def test_repeated_scoring_of_a_renormalised_case_is_exact(self):
        values = {score(case(minimality=UNKNOWN), finding(), CFG).value for _ in range(20)}
        assert values == {0.911765}


class TestPublishThreshold:
    def test_the_threshold_is_inclusive(self):
        """A value landing exactly on the configured threshold publishes. The boundary has to be
        decided somewhere and stated, or two callers will disagree about a score of 0.50."""
        borderline = case(
            sufficiency=("pass", 0.5),
            minimality=("pass", 0.5),
            holdout=("pass", 0.5),
            rival_sufficiency=0.25,
        )
        conf = score(borderline, finding(p_value=10**-3.5), CFG)
        assert conf.value == 0.5
        assert conf.value == CFG.publish_threshold
        assert conf.publishable

    def test_just_below_the_threshold_does_not_publish(self):
        borderline = case(
            sufficiency=("pass", 0.49),
            minimality=("pass", 0.5),
            holdout=("pass", 0.5),
            rival_sufficiency=0.245,
        )
        conf = score(borderline, finding(p_value=10**-3.5), CFG)
        assert conf.value == 0.497
        assert not conf.publishable

    def test_a_raised_threshold_withholds_a_case_the_default_would_publish(self):
        strict = ConfidenceConfig(publish_threshold=0.95)
        conf = score(case(), finding(), strict)
        assert conf.value == 0.91
        assert not conf.publishable

    def test_component_coverage_gates_publication_independently_of_the_value(self):
        """Lowering the threshold to zero must not let a case with one scored component
        through. The two gates guard different failures: the value says the evidence is weak,
        the count says most of the evidence was never gathered."""
        permissive = ConfidenceConfig(publish_threshold=0.0)
        alone = localization(candidate("A", observed=0.55, status="accused"))
        conf = score(alone, finding(p_value=1e-30), permissive)
        assert conf.components_scored == 1
        assert not conf.publishable


class TestWeightsAreHonoured:
    """Sufficiency is the weak component in this case, so its weight decides the direction."""

    def build(self) -> Localization:
        return case(sufficiency=("pass", 0.20), rival_sufficiency=0.04)

    def test_raising_the_weight_of_a_weak_component_lowers_the_score(self):
        heavier = ConfidenceConfig(
            weights={
                "significance": 0.10,
                "sufficiency": 0.45,
                "minimality": 0.15,
                "stability": 0.15,
                "separation": 0.15,
            }
        )
        default = score(self.build(), finding(), CFG).value
        raised = score(self.build(), finding(), heavier).value
        assert raised < default

    def test_lowering_the_weight_of_a_weak_component_raises_the_score(self):
        lighter = ConfidenceConfig(
            weights={
                "significance": 0.35,
                "sufficiency": 0.20,
                "minimality": 0.15,
                "stability": 0.15,
                "separation": 0.15,
            }
        )
        default = score(self.build(), finding(), CFG).value
        lowered = score(self.build(), finding(), lighter).value
        assert lowered > default
        assert default == pytest.approx(0.25 + 0.06 + 0.135 + 0.12 + 0.12)
        assert lowered == pytest.approx(0.35 + 0.04 + 0.135 + 0.12 + 0.12)

    def test_effective_weights_follow_the_configured_ones_through_renormalisation(self):
        heavier = ConfidenceConfig(
            weights={
                "significance": 0.10,
                "sufficiency": 0.45,
                "minimality": 0.15,
                "stability": 0.15,
                "separation": 0.15,
            }
        )
        conf = score(case(minimality=UNKNOWN), finding(), heavier)
        weights = {c.name: c.weight for c in conf.components}
        assert weights["sufficiency"] == pytest.approx(0.45 / 0.85)
        assert sum(weights.values()) == pytest.approx(1.0)


class TestExplain:
    def test_the_breakdown_shows_every_component_with_its_weight_and_reason(self):
        text = score(case(), finding(), CFG).explain()
        assert "Confidence 0.9100 of 1.0000 -- publishable." in text
        assert "5 of 5 components scored" in text
        for name in ("significance", "sufficiency", "minimality", "stability", "separation"):
            assert name in text
        assert "weight 0.3000" in text
        assert "p = 1e-06 is 6.0 decades of evidence" in text

    def test_an_unscored_component_is_shown_as_unscored_rather_than_as_zero(self):
        text = score(case(minimality=UNKNOWN), finding(), CFG).explain()
        assert "minimality   not scored   weight 0.0000" in text
        assert "Caveat:" in text
        assert "renormalised" in text

    def test_a_clean_case_carries_no_caveat(self):
        assert "Caveat:" not in score(case(), finding(), CFG).explain()

    def test_the_breakdown_stays_within_a_readable_width(self):
        """It goes into a case file next to tables and SQL, so a line that wraps in a terminal
        makes the whole file harder to scan."""
        text = score(case(sufficiency=UNKNOWN, minimality=UNKNOWN), finding(), CFG).explain()
        assert max(len(line) for line in text.splitlines()) <= 100


class TestEvidenceMustDescribeTheAccused:
    """A test describes exactly one cell, and it may only speak for that cell.

    The finding a case enters on is whichever cell had the smallest p-value in its group, which
    is routinely a two-dimensional cell *inside* the segment localization eventually names. An
    adversarial review found the consequence on live data: a case accusing `category=finance`
    was certified by a structural test on `EU x interstitial` at p = 0, and published five
    scored components with an empty caveat. It is the worst shape of error this module can
    produce, because it looks like the strongest possible case.
    """

    def other_cell(self, **kw):
        """A finding about a different segment than the one the case will accuse."""
        f = finding(**kw)
        return Finding(
            metric=f.metric,
            segment=Segment.of(country="Z"),
            window=f.window,
            detector=f.detector,
            test=f.test,
            observed_counters=f.observed_counters,
            baseline_counters=f.baseline_counters,
            phi=f.phi,
            weeks_kept=f.weeks_kept,
            weeks_seen=f.weeks_seen,
            survives_correction=f.survives_correction,
        )

    def test_a_borrowed_p_value_does_not_score_significance(self):
        got = score(case(), self.other_cell(p_value=1e-300), CFG)
        significance = next(c for c in got.components if c.name == "significance")
        assert significance.state == "unknown"

    def test_a_borrowed_finding_does_not_score_stability_either(self):
        """Weeks kept and weeks seen belong to the tested cell, not to the accused one."""
        got = score(case(), self.other_cell(), CFG)
        stability = next(c for c in got.components if c.name == "stability")
        assert stability.state == "unknown"

    def test_overwhelming_borrowed_evidence_scores_below_its_own_evidence(self):
        """The whole point: a smaller p-value from elsewhere must not raise the number."""
        borrowed = score(case(), self.other_cell(p_value=1e-300), CFG)
        own = score(case(), finding(p_value=1e-300), CFG)
        assert borrowed.value < own.value

    def test_the_caveat_names_the_cell_the_evidence_actually_describes(self):
        got = score(case(), self.other_cell(), CFG)
        assert "country=Z" in got.explain()

    def test_two_holes_leave_too_little_to_publish_on(self):
        """Significance and stability are the two the detector supplies, so borrowing loses
        both at once and only the three localization components remain."""
        got = score(case(), self.other_cell(p_value=1e-300), CFG)
        assert got.components_scored == 3
        assert not got.publishable

    def test_evidence_about_the_accused_is_still_scored_normally(self):
        got = score(case(), finding(), CFG)
        assert got.components_scored == 5
        assert got.publishable
