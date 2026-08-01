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
from verdict.detect import DetectionResult, Finding, _lattice_combos, apply_correction
from verdict.metrics import MetricRegistry
from verdict.query import Counters, Segment, Window
from verdict.schema import LATTICE_DEPTH, TOTAL_COMBO
from verdict.stats import (
    TestResult,
    pearson_dispersion,
    pearson_residuals,
    required_denominator,
    robust_dispersion,
)

REGISTRY = MetricRegistry.load(Path(__file__).resolve().parents[1] / "config" / "metrics.yaml")
WINDOW = Window(datetime(2026, 6, 23), datetime(2026, 6, 24), "1h")

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

    def needed(self, name: str) -> float:
        metric = REGISTRY.metric(name)
        return required_denominator(
            BASELINES[name], metric.effect_threshold(DetectionConfig().min_relative_effect)
        )

    @pytest.mark.parametrize("name", ["fill_rate", "render_rate", "ctr"])
    def test_the_grand_total_is_testable_within_a_day(self, name):
        have = self.available(REGISTRY.metric(name), self.DAILY, 1.0)
        assert self.needed(name) <= have, (
            f"{name} cannot be tested at the grand total in a day: needs "
            f"{self.needed(name):,.0f} but only {have:,.0f} are available"
        )

    @pytest.mark.parametrize("name", ["fill_rate", "render_rate", "ctr"])
    def test_a_median_one_way_cell_is_testable_within_a_week(self, name):
        """Localization is worthless if only the total can be tested -- the whole point is to
        name a segment."""
        assert self.needed(name) <= self.available(REGISTRY.metric(name), self.DEPTH1_MEDIAN, 7.0)

    def test_the_global_default_would_have_made_ctr_undetectable(self):
        """Pins why the per-metric override exists, so removing it fails loudly here."""
        needed = required_denominator(BASELINES["ctr"], DetectionConfig().min_relative_effect)
        assert needed > self.available(REGISTRY.metric("ctr"), self.DEPTH1_MEDIAN, 35.0)

    def test_ctr_cannot_be_localized_to_a_two_way_cell_here(self):
        """A limitation worth pinning rather than discovering during a demo. A median two-way
        cell holds ~771 impressions/day; CTR needs 29,766 even at its 25% threshold."""
        assert self.needed("ctr") > 771 * 7


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


class TestLatticeMatchesGrain:
    @pytest.mark.parametrize("grain", sorted(LATTICE_DEPTH))
    def test_never_asks_a_grain_for_depth_it_does_not_store(self, grain):
        metric = REGISTRY.metric("fill_rate")
        for combo in _lattice_combos(REGISTRY, metric, grain):
            if combo != TOTAL_COMBO:
                assert len(combo.split("|")) <= LATTICE_DEPTH[grain]

    def test_omits_dimensions_the_metric_cannot_legally_be_sliced_by(self):
        """campaign_type only exists on filled rows, so slicing fill rate by it silently
        redefines the denominator as "filled requests" and returns 1.0 everywhere."""
        combos = _lattice_combos(REGISTRY, REGISTRY.metric("fill_rate"), "1h")
        assert not any("campaign_type" in c or "vertical" in c for c in combos)

    def test_keeps_them_for_a_metric_whose_denominator_is_post_fill(self):
        combos = _lattice_combos(REGISTRY, REGISTRY.metric("ctr"), "1h")
        assert any("vertical" in c for c in combos)
