"""Tests for the statistical core.

Expected values are worked by hand from the formulas rather than captured from a previous run
of this code, so a change in behaviour fails the test instead of quietly updating the golden
value it is compared against.
"""

from __future__ import annotations

import math

import pytest

from verdict.stats import (
    _betainc,
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
    student_t_sf,
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


class TestStudentTTail:
    """Everything the ratio metrics claim rests on this tail, so it is pinned twice over:
    against distributions whose tails close in elementary functions, and against the critical
    values printed in a standard table."""

    def test_matches_the_exact_tail_on_one_degree_of_freedom(self):
        """One degree of freedom is Cauchy, whose two-sided tail is 1 - (2/pi) arctan|t|.

        Two days of history land here, and it is also where the continued fraction is asked
        for the most extreme shape parameters it will ever see. Agreement to twelve digits
        says the beta machinery is right rather than merely plausible.
        """
        for t in (0.5, 1.0, 2.0, 3.182, 12.706):
            exact = 1.0 - (2.0 / math.pi) * math.atan(t)
            assert student_t_sf(t, 1) == pytest.approx(exact, rel=1e-12)
        # t = 1 sits on the Cauchy quartiles, so exactly half the mass lies beyond it.
        assert student_t_sf(1.0, 1) == pytest.approx(0.5, rel=1e-14)

    def test_matches_the_exact_tail_on_two_degrees_of_freedom(self):
        """Two degrees of freedom closes as 1 - |t| / sqrt(2 + t^2)."""
        for t in (0.5, 1.0, 2.0, 4.303):
            exact = 1.0 - t / math.sqrt(2.0 + t * t)
            assert student_t_sf(t, 2) == pytest.approx(exact, rel=1e-12)
        assert student_t_sf(0.5, 2) == pytest.approx(2.0 / 3.0, rel=1e-14)

    def test_matches_the_exact_tail_on_three_degrees_of_freedom(self):
        """Three degrees of freedom is the operating point: four days of history, k - 1.

        Its tail closes as 1 - (2/pi)(arctan u + u/(1+u^2)) with u = t/sqrt(3). Every ratio
        p-value this system publishes is read off this curve, so it is worth pinning at the
        exact degrees of freedom rather than only in general.
        """
        for t in (0.5, 1.0, 2.0, 3.182, 5.841, 10.0):
            u = t / math.sqrt(3.0)
            exact = 1.0 - (2.0 / math.pi) * (math.atan(u) + u / (1.0 + u * u))
            assert student_t_sf(t, 3) == pytest.approx(exact, rel=1e-11)

    def test_matches_published_critical_values(self):
        """Pinned against a standard two-sided t-table, the one printed in the back of any
        statistics text.

        The tolerance here is set by the table and not by this function: critical values are
        printed to three decimals, and the df = 3 five-percent point is really 3.18245, so
        rounding it to 3.182 moves the tail by about 2e-5 on its own. The closed-form tests
        above are what pin the accuracy, to twelve digits.
        """
        table = [
            (12.706, 1, 0.05), (63.657, 1, 0.01),
            (4.303, 2, 0.05), (9.925, 2, 0.01),
            (3.182, 3, 0.05), (5.841, 3, 0.01), (2.353, 3, 0.10),
            (2.776, 4, 0.05), (4.604, 4, 0.01),
            (2.228, 10, 0.05), (3.169, 10, 0.01),
            (2.042, 30, 0.05), (2.750, 30, 0.01),
            (1.984, 100, 0.05),
        ]
        for t, df, expected in table:
            assert student_t_sf(t, df) == pytest.approx(expected, abs=5e-5)

    def test_direction_is_discarded_exactly_as_the_normal_tail_discards_it(self):
        """A fall has to score like a rise of the same size.

        Callers rank on p-value and never consult the sign, so a one-sided tail here would
        return p near 1 for every drop, and drops are most of what this system exists to
        find. `normal_sf` is two-sided and both feed the same Benjamini-Hochberg family, so
        halving one of them would also hand ratio metrics a standing advantage over
        proportions that has nothing to do with the evidence.
        """
        assert student_t_sf(-3.5, 3) == student_t_sf(3.5, 3)
        assert student_t_sf(-0.4, 7) == student_t_sf(0.4, 7)
        # 3.182 on three degrees of freedom is the 5% two-sided point, 2.5% one-sided.
        assert student_t_sf(3.182, 3) == pytest.approx(0.05, abs=5e-5)

    def test_tails_are_heavier_than_the_normal_and_tighten_as_history_grows(self):
        """The whole reason this function exists.

        On three degrees of freedom a statistic of 26 is worth about 1.2e-4. The normal tail
        calls the same number 5e-149, and that difference is exactly what let a 5% move on
        this corpus be published as p = 1.6e-155.
        """
        assert student_t_sf(26.0, 3) == pytest.approx(1.2481e-4, rel=1e-4)
        assert normal_sf(26.0) < 1e-140

        tails = [student_t_sf(3.0, df) for df in (1, 2, 3, 10, 50, 500)]
        assert tails == sorted(tails, reverse=True)
        assert tails[-1] > normal_sf(3.0)

    def test_converges_to_the_normal_tail_as_df_grows(self):
        """With enough history the correction has to cost nothing.

        If it did not converge, every additional week of baseline would leave the ratio tests
        permanently more conservative than the proportion tests they are ranked against, and
        the ranking would be measuring history length rather than evidence.
        """
        z = 1.959963985
        assert student_t_sf(z, 1_000_000) == pytest.approx(normal_sf(z), rel=1e-4)
        errors = [abs(student_t_sf(z, df) - normal_sf(z)) for df in (100, 1_000, 10_000)]
        assert errors == sorted(errors, reverse=True)
        # Convergence has to hold deep in the tail too, not only near the five-percent point.
        assert student_t_sf(8.0, 1_000_000) == pytest.approx(normal_sf(8.0), rel=1e-2)

    def test_retains_precision_far_into_the_tail(self):
        """The same property `normal_sf` needs, for the same reason.

        The elementary df = 3 form cancels to noise out here: at t = 1e5 it subtracts
        0.999999999999998 from one and keeps about two digits. The incomplete beta is
        evaluated directly rather than by subtraction, so it lands on the exact asymptote
        4*sqrt(3)/pi / t^3 to nine figures. Reporting zero instead would erase the difference
        between strong evidence and overwhelming evidence.
        """
        asymptote = 4.0 * math.sqrt(3.0) / math.pi / 1e15
        assert student_t_sf(1e5, 3) == pytest.approx(asymptote, rel=1e-8)
        assert student_t_sf(1e5, 3) > 0.0

    def test_no_statistic_is_no_evidence(self):
        assert student_t_sf(0.0, 3) == pytest.approx(1.0)

    def test_no_degrees_of_freedom_is_no_evidence(self):
        """A single historical sample estimates no scale, so there is no distribution to
        consult. Anything below 1 here would let a segment with one day of history publish."""
        assert student_t_sf(5.0, 0) == 1.0
        assert student_t_sf(5.0, -1) == 1.0

    def test_an_enormous_statistic_saturates_rather_than_raising(self):
        """Squaring the statistic with ``**`` raises OverflowError past about 1e154, which
        would take down a scan of thousands of segments over one degenerate cell. Such a
        statistic means the p-value has underflowed, which is a finding, not an error."""
        assert student_t_sf(1e200, 3) == 0.0
        assert student_t_sf(float("inf"), 3) == 0.0

    def test_large_df_does_not_overflow_the_leading_gamma(self):
        """Formed directly, the leading factor is a ratio of gamma functions, and gamma
        overflows a double past 171.6 -- around 343 degrees of freedom. An hourly baseline
        reaches that within a fortnight, and the failure would be an exception rather than a
        wrong number."""
        assert student_t_sf(2.0, 400) == pytest.approx(normal_sf(2.0), rel=0.02)
        assert student_t_sf(2.0, 100_000) == pytest.approx(normal_sf(2.0), rel=1e-4)

    def test_incomplete_beta_matches_its_elementary_special_cases(self):
        """Isolates the beta machinery from the way t is mapped onto it.

        I_x(a, 1) = x^a and I_x(1, b) = 1 - (1-x)^b are exact, and for each the two x values
        straddle the (a+1)/(a+b+2) switchover, so the continued fraction and its reflection
        are both exercised. A failure here localises to the fraction rather than to the
        parameterisation.
        """
        for x in (0.3, 0.9):
            assert _betainc(x, 5.0, 1.0) == pytest.approx(x**5.0, rel=1e-12)
            assert _betainc(x, 1.0, 5.0) == pytest.approx(1.0 - (1.0 - x) ** 5.0, rel=1e-12)
            assert _betainc(x, 1.0, 1.0) == pytest.approx(x, rel=1e-12)
        assert _betainc(0.5, 3.0, 3.0) == pytest.approx(0.5, rel=1e-12)
        assert _betainc(0.0, 2.0, 3.0) == 0.0
        assert _betainc(1.0, 2.0, 3.0) == 1.0

    def test_incomplete_beta_branches_agree_across_the_switchover(self):
        """The reflection is what lets one fast-converging expansion cover the whole range.
        If the branches disagreed, the t tail would carry a step discontinuity at whichever
        statistic maps onto the switchover, and two nearly identical segments could be
        published with p-values an order of magnitude apart."""
        for x in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
            assert _betainc(x, 1.5, 0.5) == pytest.approx(
                1.0 - _betainc(1.0 - x, 0.5, 1.5), rel=1e-12
            )


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

    def test_a_five_percent_move_against_four_tight_days_is_not_overwhelming(self):
        """The regression that motivated the whole correction.

        Four daily eCPMs inside a 0.25% band and an observation 5% below them. Dividing that
        move by a scale estimated from those same four points and reading the result off a
        normal tail gives z = -55, and the normal tail underflows to a literal p = 0.0. On the
        real nine-million-row corpus that arithmetic published eCPM and revenue-per-request at
        p = 0 and p = 1.6e-155 for falls of 5.0% and 5.1%, and those two crowded every
        fill-rate and CTR finding out of the list.

        A 5% move measured against four days is worth a couple of percent, not certainty.
        Anything below about 1e-6 here means the estimated scale is being treated as known
        again.
        """
        result = log_ratio_test(2.28, [2.4000, 2.4030, 2.3970, 2.4000])
        assert result.relative_effect == pytest.approx(-0.05)
        assert result.expected == pytest.approx(2.40)
        assert result.p_value > 1e-6
        assert result.p_value == pytest.approx(0.0195, abs=5e-4)
        assert result.direction == "fall"

    def test_a_chance_tight_spread_cannot_manufacture_certainty(self):
        """Why the degeneracy guard needed a relative floor and not only an absolute one.

        A MAD over four points is the mean of the two middle absolute deviations, so two days
        landing on the same value drag it toward zero whatever the underlying spread is. Two
        of these four are identical and the other pair differ in the fourth decimal, which
        gives a log-space scale of 3e-5, a hundredth of a percent. The absolute guard at 1e-9
        is nowhere near catching that, and Student's t alone does not save it either: the
        statistic comes to 1485 and even three degrees of freedom price that at 7e-10.

        Same 5% move as the test above, so the answer has to be the same as well.
        """
        result = log_ratio_test(2.28, [2.4000, 2.4000, 2.4001, 2.3999])
        assert result.p_value == pytest.approx(0.0195, abs=5e-4)
        assert abs(result.z) == pytest.approx(4.588, abs=0.01)

    def test_a_spread_wider_than_the_floor_is_used_as_measured(self):
        """The floor has to be a floor and nothing more.

        A segment whose history genuinely swings by several percent must be judged against
        that swing. If the floor replaced the measured scale rather than bounding it below,
        every quiet segment and every volatile one would be handed the same standard error and
        the test would stop being about the segment in front of it.
        """
        tight = log_ratio_test(2.28, [2.4000, 2.4030, 2.3970, 2.4000])
        noisy = log_ratio_test(2.28, [2.40, 2.52, 2.28, 2.40])
        assert abs(noisy.z) < abs(tight.z)
        assert noisy.p_value > 0.2

    def test_the_standard_error_carries_the_prediction_term(self):
        """What is being predicted is a new day, not the centre of the old ones.

        The centre is itself uncertain by sigma/sqrt(k), and the new observation varies by a
        further sigma around wherever the centre truly is, so the standard error is
        sigma * sqrt(1 + 1/k). Dropping the term understates the error by 12% at four samples,
        and 12% in the denominator is worth orders of magnitude out where these statistics
        land.

        Duplicating a history leaves the median and the MAD of the logs exactly where they
        were -- the middle two of eight duplicated points are the same two values as the
        middle two of four -- so k is the only thing that differs between these two calls and
        the ratio of the statistics isolates the term to the last digit.
        """
        four = log_ratio_test(3.5, [4.0, 4.1, 3.9, 4.05])
        eight = log_ratio_test(3.5, [4.0, 4.1, 3.9, 4.05] * 2)
        assert abs(eight.z) == pytest.approx(abs(four.z) * math.sqrt(1.25 / 1.125), rel=1e-12)
        assert abs(eight.z) > abs(four.z)

    def test_an_unambiguous_collapse_is_still_significant(self):
        """Heavier tails have to cost confidence, not detection.

        Four days near 2.40 and a day at 0.24 is a 90% fall. Four samples can never make that
        worth 1e-40 and it would be dishonest to say they can, but it has to stay far enough
        below any threshold an operator would set that the finding still reaches the list, and
        far enough below the 5% move that the two never merge.
        """
        history = [2.4000, 2.4030, 2.3970, 2.4000]
        collapse = log_ratio_test(0.24, history)
        assert collapse.relative_effect == pytest.approx(-0.9)
        assert collapse.p_value < 1e-6
        assert collapse.direction == "fall"
        assert collapse.p_value < log_ratio_test(2.28, history).p_value / 10_000

    def test_falls_and_rises_of_the_same_size_score_alike(self):
        """Most of what this system looks for is a metric going down. A one-sided tail would
        return p near 1 for every fall and the detector would report nothing at all."""
        history = [2.4000, 2.4030, 2.3970, 2.4000]
        fall = log_ratio_test(2.40 * 0.95, history)
        rise = log_ratio_test(2.40 / 0.95, history)
        assert fall.direction == "fall"
        assert rise.direction == "rise"
        assert fall.p_value == pytest.approx(rise.p_value, rel=1e-9)
        assert fall.p_value < 0.05

    def test_model_names_the_test_that_produced_the_number(self):
        """Case files record the model string, so a reader holding a published p-value has to
        be able to tell whether it came off a normal tail or a t one. Reusing one name for two
        different calculations makes cases from either side indistinguishable after the
        fact."""
        assert log_ratio_test(2.28, [2.4000, 2.4030, 2.3970, 2.4000]).model == "log_ratio_t"

    def test_the_spread_floor_does_not_rescue_an_exactly_flat_history(self):
        """The floor is applied after the degeneracy check rather than in place of it.

        A history flat to numerical precision carries no scale information whatsoever --
        typically fixed-price inventory, or a segment too small to have moved. Handing it the
        floor would give it a fabricated 1% spread and let it publish findings on the strength
        of a number nobody measured.
        """
        result = log_ratio_test(2.0, [4.0, 4.0, 4.0, 4.0])
        assert result.model == "log_ratio_degenerate"
        assert result.p_value == 1.0
        assert result.z == 0.0

    def test_a_non_positive_observation_is_refused_rather_than_logged(self):
        """A segment that earned nothing at all has no logarithm. The sentinel keeps it out of
        the findings list instead of raising part-way through a scan."""
        result = log_ratio_test(0.0, [2.40, 2.403, 2.397, 2.40])
        assert result.model == "log_ratio_insufficient"
        assert result.p_value == 1.0
        assert math.isnan(result.expected)


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


class TestDeclinedTestsSaySoRatherThanReportingCalm:
    """Every test returns p = 1 with a zero effect when it cannot run.

    Read without care that is indistinguishable from "this cell held steady", and a caller
    acting on it will clear a segment nobody ever measured. `testable` is what separates the
    two, and it has to hold for every path that declines, not just the one that prompted it.
    """

    def test_a_proportion_with_no_baseline_exposure_is_untestable(self):
        assert not two_proportion_test(50.0, 100.0, 0.0, 0.0).testable

    def test_a_count_with_a_zero_expectation_is_untestable(self):
        assert not count_test(500.0, 0.0).testable

    def test_a_ratio_with_too_little_history_is_untestable(self):
        assert not log_ratio_test(2.5, [3.0, 3.1]).testable

    def test_a_ratio_with_no_observation_is_untestable(self):
        assert not log_ratio_test(0.0, [3.0, 3.1, 3.05, 2.98]).testable

    def test_declining_is_not_the_same_as_finding_nothing(self):
        """The trap: both report p = 1 and no movement, so p alone cannot tell them apart."""
        declined = count_test(500.0, 0.0)
        genuine = count_test(1000.0, 1000.0)
        assert declined.p_value == genuine.p_value == 1.0
        assert declined.relative_effect == genuine.relative_effect == 0.0
        assert not declined.testable
        assert genuine.testable

    def test_real_results_are_testable(self):
        assert two_proportion_test(700.0, 1000.0, 7800.0, 10_000.0).testable
        assert count_test(1200.0, 1000.0).testable
        assert log_ratio_test(2.0, [3.0, 3.1, 3.05, 2.98]).testable
