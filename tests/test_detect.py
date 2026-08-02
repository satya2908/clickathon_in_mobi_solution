"""Tests for detection bookkeeping: the correction family and the lattice/grain contract.

The bug that motivated this file was invisible and total. Findings were filtered to
p <= alpha *before* Benjamini-Hochberg ran, so the correction only ever saw p-values already
below the threshold -- and BH keeps the largest rank k satisfying p(k) <= alpha*k/m, which for
such an input is always m. Every finding survived. The primitive was correct and unit-tested,
the call site made it an identity function, and no test covered the call site.

So these tests assert the property that matters -- that noise gets rejected -- rather than the
mechanics of the procedure, which tests/test_stats.py already pins.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from verdict.config import DetectionConfig
from verdict.detect import (
    DetectionResult,
    Finding,
    _denominator_floor,
    apply_correction,
    estimate_dispersion,
    lattice_combos,
)
from verdict.metrics import MetricRegistry
from verdict.query import Counters, Segment, Window
from verdict.schema import LATTICE_DEPTH, TOTAL_COMBO
from verdict.stats import (
    NEAR_TOTAL_COLLAPSE,
    TestResult,
    pearson_dispersion,
    pearson_residuals,
    required_denominator,
    resolvable_effect,
    robust_dispersion,
)

REGISTRY = MetricRegistry.load(Path(__file__).resolve().parents[1] / "config" / "metrics.yaml")
WINDOW = Window(datetime(2026, 6, 23), datetime(2026, 6, 24), "1h")
SEGMENT = Segment.of(country="US")

# Measured from the loaded corpus, not assumed.
BASELINES = {"fill_rate": 0.780879, "render_rate": 0.979958, "ctr": 0.010881}


def finding(
    p_value: float,
    relative_effect: float = -0.20,
    name: str = "fill_rate",
    effect_threshold: float = 0.05,
) -> Finding:
    return Finding(
        metric=name,
        effect_threshold=effect_threshold,
        segment=Segment.of(country=f"C{p_value:.12f}"),
        window=WINDOW,
        detector="temporal",
        test=TestResult(
            z=-3.0,
            p_value=p_value,
            observed=0.62,
            expected=0.78,
            absolute_effect=-0.16,
            relative_effect=relative_effect,
            model="two_proportion",
        ),
        observed_counters=Counters(requests=10_000, fills=6_200),
        baseline_counters=Counters(requests=10_000, fills=7_800),
        phi=1.0,
    )


def result_of(*findings: Finding, tested: int | None = None) -> DetectionResult:
    r = DetectionResult(findings=list(findings))
    r.tested_cells = tested if tested is not None else len(findings)
    return r


class TestCorrectionActuallyCorrects:
    def test_a_family_of_pure_noise_yields_nothing(self):
        """Under the null, p-values are uniform, so 1,700 tested cells put about 17 of them
        below 0.01 purely by chance -- the expected yield at this lattice width, and exactly
        what the old call site published as seventeen confident incidents.

        The family here is the uniform grid p_i = i/1701, which is where the order statistics
        of 1,700 uniforms sit in expectation. Nothing in it should survive.
        """
        cfg = DetectionConfig()
        m = 1700
        uniform = [finding((i + 1) / (m + 1)) for i in range(m)]
        assert sum(f.p_value <= 0.01 for f in uniform) == 17  # what the old code would report
        assert apply_correction(result_of(*uniform), cfg).findings == []

    def test_one_overwhelming_signal_survives_the_same_family(self):
        """The complement of the test above. A correction that rejects everything is equally
        useless, and would be indistinguishable from the bug if only the first test existed."""
        cfg = DetectionConfig()
        real = finding(1e-12)
        noise = [finding(0.01 + 0.99 * (i + 1) / 1700) for i in range(1699)]
        out = apply_correction(result_of(real, *noise), cfg)
        assert [f.p_value for f in out.findings] == [1e-12]

    def test_a_borderline_p_value_alone_in_a_wide_family_is_rejected(self):
        """p = 0.008 clears an uncorrected 0.01 threshold and is exactly what the old code
        published. Against 1,700 tests it is unremarkable: 0.008 > 0.01 * 1/1700."""
        cfg = DetectionConfig()
        out = apply_correction(result_of(finding(0.008), tested=1700), cfg)
        assert out.findings == []

    def test_the_same_p_value_is_reportable_in_a_family_of_one(self):
        """Whether a p-value is evidence depends on how many chances noise had. Testing one
        pre-specified cell is a different claim from scanning the lattice and reporting the
        best of it, and the correction is what encodes the difference."""
        cfg = DetectionConfig()
        out = apply_correction(result_of(finding(0.008), tested=1), cfg)
        assert len(out.findings) == 1

    def test_marks_every_finding_before_filtering(self):
        cfg = DetectionConfig()
        real, noise = finding(1e-12), finding(0.5)
        r = result_of(real, noise)
        apply_correction(r, cfg)
        assert real.survives_correction is True
        assert noise.survives_correction is False

    def test_records_that_correction_ran(self):
        """`corrected` is what stops a caller reporting from an uncorrected scan, since before
        correction `findings` is every tested cell rather than a list of anomalies."""
        r = result_of(finding(1e-12))
        assert r.corrected is False
        assert apply_correction(r, DetectionConfig()).corrected is True

    def test_an_empty_family_is_not_an_error(self):
        assert apply_correction(DetectionResult(), DetectionConfig()).findings == []


class TestEffectGateRunsAfterCorrection:
    def test_a_significant_but_trivial_move_is_not_reported(self):
        """Large denominators make tiny moves significant. A 0.4% shift is real and useless."""
        cfg = DetectionConfig(min_relative_effect=0.05)
        out = apply_correction(result_of(finding(1e-14, relative_effect=-0.004)), cfg)
        assert out.findings == []

    def test_small_effects_still_count_toward_the_family_size(self):
        """The ordering that fixes the bug, shown as a difference in outcome.

        The same borderline finding is reportable alone and rejected inside a wide family. If
        the effect gate ran first, the 1,699 trivial-effect cells would be gone before the
        correction saw them, m would collapse to 1, and the borderline cell would be published
        -- which is the permissive direction, and the reason ordering matters here.
        """
        cfg = DetectionConfig(min_relative_effect=0.05)
        borderline = finding(0.004, relative_effect=-0.20)
        trivial = [finding((i + 1) / 1700, relative_effect=-0.001) for i in range(1699)]

        assert len(apply_correction(result_of(borderline), cfg).findings) == 1
        assert apply_correction(result_of(borderline, *trivial), cfg).findings == []


class TestFamilyIsPooledAcrossMetrics:
    def test_pooling_two_metrics_is_stricter_than_correcting_each(self):
        """Ten metrics scanning the same lattice is ten times the chances, so correcting each
        one separately leaves the overall error rate multiplied by ten -- the same mistake as
        correcting per combo, one level up."""
        cfg = DetectionConfig()
        a = result_of(*[finding(0.002 + 1e-9 * i, name="fill_rate") for i in range(400)])
        b = result_of(*[finding(0.002 + 1e-9 * i, name="ctr") for i in range(400)])

        pooled = DetectionResult()
        pooled.extend(a)
        pooled.extend(b)
        together = len(apply_correction(pooled, cfg).findings)

        separate = len(apply_correction(a, cfg).findings) + len(apply_correction(b, cfg).findings)
        assert together <= separate

    def test_extend_does_not_claim_correction_it_did_not_perform(self):
        corrected = apply_correction(result_of(finding(1e-12)), DetectionConfig())
        pooled = DetectionResult()
        pooled.extend(corrected)
        pooled.extend(result_of(finding(0.5)))
        assert pooled.corrected is False


class TestEveryMetricIsActuallyDetectable:
    """Guards against a threshold that silently retires a metric.

    A `min_relative_effect` too small for a metric's baseline rate does not make the system
    more sensitive -- it makes the required sample size unreachable, so every cell is filed as
    a coverage gap and no incident on that metric can ever be reported. The failure is silent
    and looks like a clean bill of health.

    The traffic figures below are measured from the loaded corpus: 9M requests over 35 days,
    196,773 impressions/day, and a median country cell holding 9,732 impressions/day.
    """

    DAILY = {"requests": 257_143, "fills": 200_754, "impressions": 196_773}
    DEPTH1_MEDIAN = {"requests": 12_723, "fills": 9_934, "impressions": 9_732}

    def available(self, metric, per_day: dict[str, int], days: float) -> float:
        return per_day[metric.denominator_field] * days

    def floor(self, name: str) -> float:
        """What the detection floor actually is: sized against a near-total collapse."""
        return required_denominator(BASELINES[name], NEAR_TOTAL_COLLAPSE)

    @pytest.mark.parametrize("name", ["fill_rate", "render_rate", "ctr"])
    def test_a_median_one_way_cell_clears_the_floor_in_a_day(self, name):
        have = self.available(REGISTRY.metric(name), self.DEPTH1_MEDIAN, 1.0)
        assert self.floor(name) <= have, (
            f"{name} cannot be tested at all in a median one-way cell: the floor is "
            f"{self.floor(name):,.0f} but only {have:,.0f} are available"
        )

    @pytest.mark.parametrize("name", ["fill_rate", "render_rate", "ctr"])
    def test_the_floor_does_not_depend_on_the_reporting_threshold(self, name):
        """The property that removes the overfitting.

        The floor used to be sized from `min_relative_effect`, so keeping detection alive meant
        choosing that constant to suit the traffic in the corpus at hand -- which is fitting a
        hyperparameter to the data. It is now sized from the edge of the possible instead, so
        changing the reporting policy cannot switch detection off, and no value in the config
        needs to know how much traffic this particular dataset happens to carry.
        """
        history = {SEGMENT: [Counters()] + [self.counters(name)] * 4}
        floors = {
            _denominator_floor(
                REGISTRY.metric(name),
                history,
                DetectionConfig(min_relative_effect=thr),
                1.0,
                baseline_weeks=4,
            )
            for thr in (0.01, 0.05, 0.25, 0.50)
        }
        assert len(floors) == 1

    def counters(self, name: str) -> Counters:
        rate = BASELINES[name]
        if name == "fill_rate":
            return Counters(requests=100_000, fills=int(100_000 * rate))
        if name == "render_rate":
            return Counters(requests=100_000, fills=100_000, impressions=int(100_000 * rate))
        return Counters(
            requests=100_000, fills=100_000, impressions=100_000, clicks=int(100_000 * rate)
        )

    @pytest.mark.parametrize("name", ["fill_rate", "render_rate", "ctr"])
    def test_sensitivity_is_reported_rather_than_assumed(self, name):
        """Every tested cell says what it could have seen, so a null result is quantified."""
        have = self.available(REGISTRY.metric(name), self.DEPTH1_MEDIAN, 1.0)
        effect = resolvable_effect(BASELINES[name], have)
        assert effect is not None and 0.0 < effect < NEAR_TOTAL_COLLAPSE

    def test_a_cell_too_small_for_any_effect_reports_none(self):
        assert resolvable_effect(BASELINES["ctr"], 50) is None

    def test_ctr_is_far_less_sensitive_than_fill_rate_at_equal_traffic(self):
        """Not a threshold to tune but a fact about the arithmetic: variance per sample goes as
        p(1-p) while the effect goes as p, so a low rate costs sensitivity."""
        fill = resolvable_effect(BASELINES["fill_rate"], 50_000)
        ctr = resolvable_effect(BASELINES["ctr"], 50_000)
        assert fill is not None and ctr is not None
        assert ctr > 5 * fill


class TestDispersionSurvivesAContaminatedBaseline:
    """The bug that made every floor unreachable.

    One baseline week in this corpus carries a real global fill-rate incident: weeks 2, 3 and 4
    sit at 0.785 for essentially every country while week 1 sits between 0.70 and 0.77. The
    mean-based Pearson estimate turned that into phi between 25 and 128 depending on the
    segment, pegged against the ceiling of 50. Every denominator floor scales with phi, so a
    fifty-fold inflation put every cell in the lattice below the floor and the detector reported
    nothing at all for a day on which fill rate had visibly moved.
    """

    def clean(self, weeks: int = 4, n: int = 20_000, p: float = 0.78) -> list[tuple]:
        return [(n, n * p, p)] * weeks

    def contaminated(self, n: int = 20_000, p: float = 0.78, bad: float = 0.70,
                     *, robust_centre: bool = True) -> list[tuple]:
        """Three weeks at the true rate and one carrying an incident, as measured here.

        `robust_centre` selects how the group's reference rate was derived: the median of the
        weekly rates, as estimate_dispersion now does, or the pooled mean it used to use. The
        distinction is the point -- a robust spread around a contaminated centre is still wrong.
        """
        centre = p if robust_centre else (3 * p + bad) / 4
        return [(n, n * p, centre)] * 3 + [(n, n * bad, centre)]

    def test_the_mean_based_estimate_is_destroyed_by_one_bad_week(self):
        phi = pearson_dispersion(self.contaminated(robust_centre=False), n_groups=1)
        assert phi > 50, f"expected the failure to reproduce, got phi={phi:.1f}"

    def test_a_robust_spread_around_a_pooled_centre_is_not_enough(self):
        """Pooling the reference rate over all four weeks drags it toward the incident, so the
        three good weeks each show a large residual and the median no longer rescues it."""
        phi = robust_dispersion(pearson_residuals(self.contaminated(robust_centre=False)))
        assert phi > 50

    def test_the_robust_estimate_is_not(self):
        phi = robust_dispersion(pearson_residuals(self.contaminated()))
        assert phi < 5.0, f"one contaminated week still dominates: phi={phi:.1f}"

    def test_both_agree_when_nothing_is_contaminated(self):
        """The robust estimator must not simply return a small number regardless. On clean data
        it has to land near the mean-based value, or it is not measuring dispersion at all."""
        cells = [(20_000, 20_000 * r, 0.78) for r in (0.770, 0.776, 0.784, 0.790)]
        mean_based = pearson_dispersion(cells, n_groups=1)
        assert robust_dispersion(pearson_residuals(cells)) == pytest.approx(
            mean_based, rel=0.75
        )

    def test_genuine_overdispersion_is_still_reported(self):
        """A metric that really does vary far more than binomial must still inflate the floor --
        suppressing that would make every test overconfident."""
        cells = [(20_000, 20_000 * r, 0.78) for r in (0.70, 0.74, 0.82, 0.86)]
        assert robust_dispersion(pearson_residuals(cells)) > 100

    def test_no_usable_residuals_falls_back_to_one(self):
        assert robust_dispersion([]) == 1.0
        assert robust_dispersion([float("nan"), float("inf")]) == 1.0


class TestUnmeasurableDispersionIsNotAssumedAway:
    """One is not a neutral default for an unmeasured variance.

    A phi of 1.0 asserts the metric is exactly as well behaved as a textbook binomial variable
    with no unmodelled structure whatsoever, which is the strongest claim the estimator can
    make rather than the weakest. Returning it when there was nothing to measure meant windows
    near the start of the data -- where a weekly baseline has at most one aligned sample -- were
    judged against a variance nobody had computed. Measured over eight planted-free days that
    produced confident accusations at two per day, one of them a 13.9% move priced at 1e-223.
    """

    def counters(self, rate: float = 0.78) -> Counters:
        return Counters(requests=100_000, fills=int(100_000 * rate))

    def history(self, weeks: int) -> dict:
        # weeks[0] is the window under investigation; the rest is the baseline it is judged on.
        return {SEGMENT: [self.counters()] + [self.counters()] * weeks}

    def test_a_single_baseline_week_cannot_estimate_dispersion(self):
        metric = REGISTRY.metric("fill_rate")
        assert estimate_dispersion(self.history(1), metric, DetectionConfig()) is None

    def test_two_baseline_weeks_are_enough_because_segments_pool(self):
        """One residual per segment says nothing alone and everything in aggregate."""
        metric = REGISTRY.metric("fill_rate")
        assert estimate_dispersion(self.history(2), metric, DetectionConfig()) is not None

    def test_no_history_at_all_is_unmeasurable_rather_than_well_behaved(self):
        metric = REGISTRY.metric("fill_rate")
        assert estimate_dispersion({}, metric, DetectionConfig()) is None


class TestLatticeMatchesGrain:
    @pytest.mark.parametrize("grain", sorted(LATTICE_DEPTH))
    def test_never_asks_a_grain_for_depth_it_does_not_store(self, grain):
        metric = REGISTRY.metric("fill_rate")
        for combo in lattice_combos(REGISTRY, metric, grain):
            if combo != TOTAL_COMBO:
                assert len(combo.split("|")) <= LATTICE_DEPTH[grain]

    def test_omits_dimensions_the_metric_cannot_legally_be_sliced_by(self):
        """campaign_type only exists on filled rows, so slicing fill rate by it silently
        redefines the denominator as "filled requests" and returns 1.0 everywhere."""
        combos = lattice_combos(REGISTRY, REGISTRY.metric("fill_rate"), "1h")
        assert not any("campaign_type" in c or "vertical" in c for c in combos)

    def test_keeps_them_for_a_metric_whose_denominator_is_post_fill(self):
        combos = lattice_combos(REGISTRY, REGISTRY.metric("ctr"), "1h")
        assert any("vertical" in c for c in combos)
