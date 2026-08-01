"""Tests for the statistical core.

Expected values are worked by hand from the formulas rather than captured from a previous run
of this code, so a change in behaviour fails the test instead of quietly updating the golden
value it is compared against.
"""

from __future__ import annotations

import math

import pytest

from verdict.stats import (
    benjamini_hochberg,
    clamp_dispersion,
    count_test,
    log_ratio_test,
    mad,
    median,
    normal_sf,
    pearson_dispersion,
    pooled_rate,
    required_denominator,
    trim_and_pool,
    two_proportion_test,
    wilson_interval,
)


class TestPooling:
    def test_pooled_rate_is_volume_weighted_not_a_mean_of_rates(self):
        """The distinction that makes rollups correct.

        One week of 1,000 requests at 50% and one of 9,000 at 90% pools to 86%, not to the
        70% an average of the two rates would give. Getting this wrong biases every baseline
        toward whichever week was quietest.
        """
        samples = [(1000.0, 500.0), (9000.0, 8100.0)]
        assert pooled_rate(samples) == pytest.approx(8600 / 10000)
        naive = (0.5 + 0.9) / 2
        assert pooled_rate(samples) != pytest.approx(naive)

    def test_trim_drops_the_week_furthest_from_the_median_rate(self):
        """A contaminated week must not reach the baseline.

        Three weeks near 80% and one at 20%: the outlier is dropped and the pooled rate comes
        from the survivors only.
        """
        samples = [(1000.0, 800.0), (1000.0, 790.0), (1000.0, 810.0), (1000.0, 200.0)]
        pooled = trim_and_pool(samples)
        assert pooled.weeks_seen == 4
        assert pooled.weeks_kept == 3
        assert pooled.dropped_rate == pytest.approx(0.2)
        assert pooled.rate == pytest.approx(2400 / 3000)

    def test_trim_judges_extremeness_on_rate_not_volume(self):
        """An unusually large week with a normal rate is good evidence and must survive."""
        samples = [(1000.0, 800.0), (1000.0, 800.0), (50000.0, 40000.0), (1000.0, 500.0)]
        pooled = trim_and_pool(samples)
        assert pooled.dropped_rate == pytest.approx(0.5)
        assert pooled.n == pytest.approx(52000.0)

    def test_no_trim_below_three_samples(self):
        """Trimming one of two leaves a single week pretending to be a robust baseline."""
        pooled = trim_and_pool([(1000.0, 800.0), (1000.0, 200.0)])
        assert pooled.weeks_kept == 2
        assert pooled.dropped_rate is None

    def test_empty_history_is_unusable_rather_than_zero(self):
        pooled = trim_and_pool([])
        assert not pooled.usable
        assert math.isnan(pooled.rate)

    def test_zero_denominator_weeks_are_ignored(self):
        pooled = trim_and_pool([(0.0, 0.0), (1000.0, 800.0), (1000.0, 820.0)])
        assert pooled.weeks_seen == 2
        assert pooled.rate == pytest.approx(1620 / 2000)


class TestSignificance:
    def test_two_proportion_z_matches_hand_calculation(self):
        """0.706 against a four-week baseline of 0.785.

        p_pool = 3846/5000 = 0.7692
        var    = 0.7692 * 0.2308 * (1/1000 + 1/4000) = 2.21915e-4
        z      = -0.079 / 0.0148967 = -5.303
        """
        result = two_proportion_test(706, 1000, 3140, 4000)
        assert result.z == pytest.approx(-5.303, abs=0.01)
        assert result.observed == pytest.approx(0.706)
        assert result.expected == pytest.approx(0.785)
        assert result.relative_effect == pytest.approx(-0.100636, abs=1e-5)
        assert result.direction == "fall"

    def test_dispersion_widens_the_test(self):
        """Overdispersion must reduce confidence, never increase it."""
        plain = two_proportion_test(706, 1000, 3140, 4000, phi=1.0)
        inflated = two_proportion_test(706, 1000, 3140, 4000, phi=4.0)
        assert abs(inflated.z) == pytest.approx(abs(plain.z) / 2.0, rel=1e-9)
        assert inflated.p_value > plain.p_value

    def test_baseline_uncertainty_is_carried(self):
        """A one-week baseline must be less conclusive than a four-week one at the same rate."""
        thin = two_proportion_test(706, 1000, 785, 1000)
        thick = two_proportion_test(706, 1000, 3140, 4000)
        assert abs(thin.z) < abs(thick.z)

    def test_degenerate_pooled_rate_is_silent(self):
        """Every request filled in both windows: no variance, so no finding."""
        result = two_proportion_test(1000, 1000, 4000, 4000)
        assert result.z == 0.0
        assert result.p_value == 1.0

    def test_rises_are_detected_symmetrically(self):
        fall = two_proportion_test(706, 1000, 3140, 4000)
        rise = two_proportion_test(864, 1000, 3140, 4000)
        assert fall.direction == "fall"
        assert rise.direction == "rise"
        assert rise.z > 0

    def test_zero_denominator_yields_no_finding(self):
        assert two_proportion_test(0, 0, 3140, 4000).p_value == 1.0

    def test_count_test_is_quasi_poisson(self):
        """z = (900 - 1000) / sqrt(4 * 1000) = -1.5811"""
        result = count_test(900, 1000, phi=4.0)
        assert result.z == pytest.approx(-1.5811, abs=1e-3)
        assert result.relative_effect == pytest.approx(-0.1)


class TestNormalTail:
    def test_two_sided_tail_at_known_points(self):
        assert normal_sf(1.959963985) == pytest.approx(0.05, abs=1e-6)
        assert normal_sf(2.575829304) == pytest.approx(0.01, abs=1e-6)
        assert normal_sf(0.0) == pytest.approx(1.0)

    def test_retains_precision_far_into_the_tail(self):
        """Genuine incidents reach z beyond 10. Subtracting from one would underflow to zero
        and erase the difference between strong and overwhelming evidence."""
        assert normal_sf(10.0) > 0.0
        assert normal_sf(10.0) == pytest.approx(1.5239e-23, rel=1e-3)
        assert normal_sf(20.0) > 0.0

    def test_symmetric_in_sign(self):
        assert normal_sf(-3.5) == normal_sf(3.5)


class TestDispersion:
    def test_pure_binomial_data_gives_phi_near_one(self):
        cells = [(1000.0, 800.0, 0.8)] * 10
        # Every cell sits exactly on its group rate, so the residual sum is zero.
        assert pearson_dispersion(cells, n_groups=1) == pytest.approx(0.0)

    def test_n_minus_g_correction_raises_the_estimate(self):
        """Dividing by N rather than N-G understates dispersion. With 8 cells and 4 groups the
        corrected estimate must be exactly twice the naive one."""
        cells = [
            (1000.0, 820.0, 0.8), (1000.0, 780.0, 0.8),
            (1000.0, 830.0, 0.8), (1000.0, 770.0, 0.8),
            (1000.0, 815.0, 0.8), (1000.0, 785.0, 0.8),
            (1000.0, 825.0, 0.8), (1000.0, 775.0, 0.8),
        ]
        corrected = pearson_dispersion(cells, n_groups=4)
        naive = pearson_dispersion(cells, n_groups=0)
        assert corrected == pytest.approx(naive * 8 / 4, rel=1e-9)

    def test_insufficient_degrees_of_freedom_falls_back_to_one(self):
        assert pearson_dispersion([(1000.0, 800.0, 0.8)], n_groups=5) == 1.0

    def test_clamp_rejects_sub_binomial_estimates(self):
        """Below 1.0 means segments are negatively correlated, which ad serving is not.
        Honouring it would make every test more confident than the data supports."""
        assert clamp_dispersion(0.75) == 1.0
        assert clamp_dispersion(3.2) == 3.2
        assert clamp_dispersion(900.0, ceiling=50.0) == 50.0
        assert clamp_dispersion(float("nan")) == 1.0


class TestPowerFloors:
    def test_ctr_needs_orders_of_magnitude_more_traffic_than_fill_rate(self):
        """The measurement that makes a single global volume floor indefensible.

        At a 10% relative drop, fill rate near 0.785 needs ~717 samples while CTR near 0.02
        needs ~109,000. Any one constant is simultaneously far too lax for one and far too
        strict for the other, and the lax side produces confident findings on segments that
        never had the traffic to support them.
        """
        fill = required_denominator(0.785, 0.10)
        ctr = required_denominator(0.02, 0.10)
        assert fill == pytest.approx(717, rel=0.02)
        assert ctr == pytest.approx(108_800, rel=0.02)
        assert ctr / fill > 100

    def test_smaller_effects_need_roughly_quadratically_more_data(self):
        """Halving the effect costs close to four times the traffic, but not exactly.

        The 1/(p1-p2)^2 term contributes the factor of four; the variance terms in the
        numerator shrink slightly as p2 moves back toward p1, which pulls the ratio down to
        about 3.8. Asserting exactly four would be asserting an approximation the formula
        does not actually make.
        """
        ratio = required_denominator(0.785, 0.05) / required_denominator(0.785, 0.10)
        assert 3.5 < ratio < 4.0

    def test_dispersion_scales_the_requirement(self):
        assert required_denominator(0.785, 0.10, phi=4.0) == pytest.approx(
            required_denominator(0.785, 0.10) * 4.0, rel=1e-9
        )

    def test_impossible_effects_are_infinite(self):
        assert required_denominator(0.5, 3.0) == float("inf")
        assert required_denominator(0.0, 0.1) == float("inf")


class TestLogRatio:
    def test_detects_a_shift_against_the_segments_own_history(self):
        result = log_ratio_test(2.0, [4.0, 4.1, 3.9, 4.05, 3.95])
        assert result.expected == pytest.approx(4.0, abs=0.06)
        assert result.relative_effect < -0.4
        assert abs(result.z) > 5

    def test_flat_history_is_untestable_not_infinitely_confident(self):
        """Zero spread usually means a segment too small to have moved, not certainty."""
        result = log_ratio_test(2.0, [4.0, 4.0, 4.0, 4.0])
        assert result.model == "log_ratio_degenerate"
        assert result.p_value == 1.0

    def test_short_history_is_refused(self):
        assert log_ratio_test(2.0, [4.0, 4.1]).model == "log_ratio_insufficient"

    def test_one_prior_outlier_does_not_hide_the_current_one(self):
        """A MAD keeps the interval tight where a standard deviation would be inflated by the
        very contamination the baseline is supposed to resist."""
        history = [4.0, 4.1, 3.9, 4.05, 12.0]
        assert abs(log_ratio_test(2.0, history).z) > 5


class TestWilson:
    def test_stays_within_zero_and_one_for_extreme_rates(self):
        low, high = wilson_interval(1, 10)
        assert 0.0 <= low < high <= 1.0
        low, high = wilson_interval(0, 5)
        assert low == 0.0 and high < 1.0

    def test_interval_narrows_as_evidence_grows(self):
        small = wilson_interval(80, 100)
        large = wilson_interval(8000, 10000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_no_data_means_no_information(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestMultipleComparisons:
    def test_uncorrected_thresholds_would_pass_noise(self):
        """One thousand null tests produce about ten p-values below 0.01 by construction.
        Benjamini-Hochberg must reject all of them."""
        p_values = [(i + 0.5) / 1000 for i in range(1000)]
        keep = benjamini_hochberg(p_values, alpha=0.01)
        assert sum(keep) == 0
        assert sum(1 for p in p_values if p <= 0.01) >= 9

    def test_genuine_signal_survives_among_noise(self):
        p_values = [1e-12, 1e-11, 1e-10] + [(i + 0.5) / 1000 for i in range(997)]
        keep = benjamini_hochberg(p_values, alpha=0.01)
        assert keep[0] and keep[1] and keep[2]
        assert sum(keep) == 3

    def test_empty_input(self):
        assert benjamini_hochberg([]) == []


class TestRobustCentre:
    def test_median_of_even_and_odd_lengths(self):
        assert median([3.0, 1.0, 2.0]) == 2.0
        assert median([4.0, 1.0, 3.0, 2.0]) == 2.5

    def test_mad_is_scaled_to_compare_with_a_standard_deviation(self):
        assert mad([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.4826, abs=1e-4)

    def test_mad_of_constant_series_is_zero(self):
        assert mad([7.0, 7.0, 7.0]) == 0.0

    def test_median_of_empty_raises(self):
        with pytest.raises(ValueError):
            median([])
