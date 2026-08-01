"""Tests for the synthetic incident injector.

The injector is the instrument every accuracy claim is measured with, so a bug here does not
produce a wrong answer, it produces a wrong *score* -- and a wrong score is believed. Three
properties carry most of the weight:

  * The funnel invariant holds after every shape. An event set describing impressions that were
    never filled is not a slightly damaged corpus, it is a corpus where every rollup is
    internally consistent and wrong.
  * The interaction really is invisible in both marginals, and the compensating pair really
    does leave the grand total alone. Those two shapes exist to be hard; if the injector
    quietly turns them into main effects, the detectors get credit for catching something they
    were never asked to see.
  * The control changes nothing at all, row for row. It is the only measurement of the false
    positive rate.

Metric values are recomputed here from raw counters rather than through the injector's own
``metric_value``. Sharing the arithmetic would let one wrong denominator satisfy both the
injection and the assertion.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from verdict.inject import (
    KIND_CLEAN,
    KIND_COMPENSATING_PAIR,
    KIND_HIGH_CARDINALITY,
    KIND_INTERACTION,
    KIND_MAIN_EFFECT,
    KIND_SLOW_DRIFT,
    KIND_SPIKE,
    KINDS,
    DimensionIndex,
    IncidentSpec,
    InjectionError,
    apply,
    apply_rows,
    plan,
)

T0 = datetime(2026, 6, 1)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

COLUMNS = (
    "event_time",
    "app_id",
    "advertiser_id",
    "geo_device_id",
    "country",
    "device_model",
    "ad_format",
    "is_filled",
    "is_impression",
    "is_click",
    "revenue",
)

ARROW_SCHEMA = pa.schema(
    [
        ("event_time", pa.timestamp("ms")),
        ("app_id", pa.string()),
        ("advertiser_id", pa.string()),
        ("geo_device_id", pa.string()),
        ("country", pa.string()),
        ("device_model", pa.string()),
        ("ad_format", pa.string()),
        ("is_filled", pa.uint8()),
        ("is_impression", pa.uint8()),
        ("is_click", pa.uint8()),
        ("revenue", pa.float64()),
    ]
)


def spread(n: int, rate: float) -> list[int]:
    """``n`` flags with the ones distributed evenly rather than clumped at the front.

    Clumping would make every test result depend on where in the window the injector happened
    to select from, which is precisely the bias the injector's uniform sampling exists to
    avoid. An evenly spread baseline means a drift slice or a spike window carries the same
    rate as the whole corpus.
    """
    out = []
    accumulated = 0.0
    for _ in range(n):
        accumulated += rate
        if accumulated >= 1.0 - 1e-9:
            out.append(1)
            accumulated -= 1.0
        else:
            out.append(0)
    return out


def cell(
    n: int,
    *,
    start: datetime = T0,
    span: timedelta = HOUR,
    fill: float = 0.78,
    render: float = 0.98,
    ctr: float = 0.02,
    revenue: float = 0.02,
    **dims: str,
) -> list[dict]:
    """One homogeneous block of events with exactly the requested funnel rates."""
    filled = spread(n, fill)
    impressions = spread(sum(filled), render)
    clicks = spread(sum(impressions), ctr)
    seconds = span.total_seconds()

    rows: list[dict] = []
    impression_at = 0
    click_at = 0
    for i in range(n):
        is_filled = filled[i]
        is_impression = 0
        is_click = 0
        if is_filled:
            is_impression = impressions[impression_at]
            impression_at += 1
            if is_impression:
                is_click = clicks[click_at]
                click_at += 1
        row = {
            "event_time": start + timedelta(seconds=seconds * i / n),
            "app_id": "app_00",
            "advertiser_id": f"adv_{i % 3}" if is_filled else "",
            "geo_device_id": "gd_00",
            "country": "XX",
            "device_model": "Model A",
            "ad_format": "banner",
            "is_filled": is_filled,
            "is_impression": is_impression,
            "is_click": is_click,
            "revenue": revenue if is_impression else 0.0,
        }
        row.update(dims)
        rows.append(row)
    return rows


def combine(*groups: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for group in groups:
        rows.extend(group)
    rows.sort(key=lambda r: r["event_time"])
    return rows


def select(
    rows: list[dict],
    where: dict | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict]:
    out = rows
    if where:
        out = [r for r in out if all(r.get(k) == v for k, v in where.items())]
    if since is not None:
        out = [r for r in out if r["event_time"] >= since]
    if until is not None:
        out = [r for r in out if r["event_time"] < until]
    return out


def measure(rows: list[dict], metric: str) -> float | None:
    """The metric, restated from counters independently of the module under test."""
    requests = len(rows)
    fills = sum(r["is_filled"] for r in rows)
    impressions = sum(r["is_impression"] for r in rows)
    clicks = sum(r["is_click"] for r in rows)
    revenue = sum(r["revenue"] for r in rows)
    pairs = {
        "fill_rate": (fills, requests, 1.0),
        "render_rate": (impressions, fills, 1.0),
        "ctr": (clicks, impressions, 1.0),
        "ecpm": (revenue, impressions, 1000.0),
    }
    numerator, denominator, scale = pairs[metric]
    return numerator / denominator * scale if denominator else None


def shift(before: list[dict], after: list[dict], metric: str) -> float:
    was = measure(before, metric)
    now = measure(after, metric)
    assert was, f"{metric} is undefined or zero before injection; the test proves nothing"
    return (now - was) / was


def assert_funnel(rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        assert r["is_impression"] <= r["is_filled"], f"row {i} rendered without a fill"
        assert r["is_click"] <= r["is_impression"], f"row {i} clicked without an impression"
        assert r["is_impression"] or r["revenue"] == 0.0, f"row {i} earned without an impression"


def millis(value: datetime) -> int:
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return int((aware - EPOCH).total_seconds() * 1000)


def touched(rows: list[dict], incident) -> set[int]:
    """Indices the plan is allowed to change: in one of its segments and one of its windows."""
    out: set[int] = set()
    for edit in incident.edits:
        for i, row in enumerate(rows):
            if not edit.start_ms <= millis(row["event_time"]) < edit.end_ms:
                continue
            if all(row.get(k) == v for k, v in edit.where):
                out.add(i)
    return out


def to_arrow(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist([{k: r[k] for k in COLUMNS} for r in rows], schema=ARROW_SCHEMA)


# Each scenario is a corpus paired with the plan it was built to exercise. The shape-specific
# tests and the invariant tests read the same pairs, so a corpus can never be quietly tuned to
# make one assertion pass while another shape goes untested.


def scenario_main_effect():
    rows = combine(
        *(
            cell(2400, span=DAY, country=c, geo_device_id=f"gd_{c}")
            for c in ("AR", "BR", "US", "IN")
        )
    )
    incident = plan(
        IncidentSpec(
            kind=KIND_MAIN_EFFECT,
            metric="fill_rate",
            where={"country": "AR"},
            start=T0 + 6 * HOUR,
            end=T0 + 18 * HOUR,
            magnitude=-0.35,
            seed=11,
        )
    )
    return rows, incident


def scenario_interaction():
    countries = [f"C{i}" for i in range(8)]
    devices = [f"D{i}" for i in range(8)]
    rows = combine(
        *(
            cell(200, span=6 * HOUR, country=c, device_model=d)
            for c in countries
            for d in devices
        )
    )
    incident = plan(
        IncidentSpec(
            kind=KIND_INTERACTION,
            metric="fill_rate",
            where={"country": "C3", "device_model": "D5"},
            start=T0,
            end=T0 + 6 * HOUR,
            magnitude=-0.40,
            seed=5,
        )
    )
    return rows, incident


def scenario_compensating_pair():
    # The counterpart deliberately runs at a much lower fill rate over more traffic, because
    # cancelling in counted units means it has to have enough unfilled requests to absorb every
    # fill the primary segment gives up.
    rows = combine(
        cell(2000, span=6 * HOUR, country="AR"),
        cell(4000, span=6 * HOUR, fill=0.50, country="BR"),
        cell(1000, span=6 * HOUR, country="US"),
    )
    incident = plan(
        IncidentSpec(
            kind=KIND_COMPENSATING_PAIR,
            metric="fill_rate",
            where={"country": "AR"},
            counterpart={"country": "BR"},
            start=T0,
            end=T0 + 6 * HOUR,
            magnitude=-0.30,
            seed=3,
        )
    )
    return rows, incident


def scenario_high_cardinality():
    rows = combine(
        *(cell(1500, span=6 * HOUR, app_id=f"app_{i:02d}") for i in range(6))
    )
    incident = plan(
        IncidentSpec(
            kind=KIND_HIGH_CARDINALITY,
            metric="fill_rate",
            where={"app_id": "app_03"},
            start=T0,
            end=T0 + 6 * HOUR,
            magnitude=-0.30,
            seed=17,
        )
    )
    return rows, incident


def scenario_slow_drift():
    rows = cell(5000, span=5 * DAY, country="AR")
    incident = plan(
        IncidentSpec(
            kind=KIND_SLOW_DRIFT,
            metric="fill_rate",
            where={"country": "AR"},
            start=T0,
            end=T0 + 5 * DAY,
            magnitude=-0.25,
            seed=23,
        )
    )
    return rows, incident


def scenario_spike():
    rows = cell(14_400, span=DAY, country="AR")
    incident = plan(
        IncidentSpec(
            kind=KIND_SPIKE,
            metric="fill_rate",
            where={"country": "AR"},
            start=T0 + 10 * HOUR,
            end=T0 + 10 * HOUR + timedelta(minutes=5),
            magnitude=-0.60,
            seed=29,
        )
    )
    return rows, incident


def scenario_clean():
    rows, _ = scenario_main_effect()
    incident = plan(
        IncidentSpec(
            kind=KIND_CLEAN,
            metric="fill_rate",
            where={},
            start=T0,
            end=T0 + DAY,
            magnitude=0.0,
            seed=1,
        )
    )
    return rows, incident


SCENARIOS = {
    KIND_MAIN_EFFECT: scenario_main_effect,
    KIND_INTERACTION: scenario_interaction,
    KIND_COMPENSATING_PAIR: scenario_compensating_pair,
    KIND_HIGH_CARDINALITY: scenario_high_cardinality,
    KIND_SLOW_DRIFT: scenario_slow_drift,
    KIND_SPIKE: scenario_spike,
    KIND_CLEAN: scenario_clean,
}


class TestFunnelInvariant:
    """Every shape, every time. A funnel violation is the one failure mode that survives every
    downstream sanity check, because the rollups built from impossible events add up
    perfectly."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_shape_leaves_the_funnel_consistent(self, kind):
        rows, incident = SCENARIOS[kind]()
        assert_funnel(rows)
        out, report = apply_rows(rows, incident)
        assert_funnel(out)
        assert report.funnel_violations == 0

    def test_clearing_a_fill_also_clears_everything_downstream(self):
        """The specific cascade. A row that stops being filled but keeps its impression, click
        and revenue is the exact shape of the bug this guards."""
        rows = cell(400, country="AR", fill=1.0, render=1.0, ctr=1.0)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0,
                end=T0 + HOUR,
                magnitude=-0.5,
                seed=2,
            )
        )
        out, report = apply_rows(rows, incident)
        cleared = [r for r in out if not r["is_filled"]]
        assert len(cleared) == 200
        assert all(r["is_impression"] == 0 for r in cleared)
        assert all(r["is_click"] == 0 for r in cleared)
        assert all(r["revenue"] == 0.0 for r in cleared)
        assert report.funnel_violations == 0

    def test_the_audit_can_actually_see_a_violation(self):
        """A positive control for the check itself.

        Every other test in this class asserts that the audit found nothing, and an audit that
        always finds nothing satisfies all of them. This feeds it an event that could not have
        happened and requires it to notice, and to say the injection did not cause it.
        """
        rows = cell(200, country="AR")
        rows[7]["is_impression"] = 1
        rows[7]["is_filled"] = 0
        rows[11]["revenue"] = 0.05
        rows[11]["is_impression"] = 0

        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="ctr",
                where={"country": "AR"},
                start=T0,
                end=T0 + HOUR,
                magnitude=-0.5,
                seed=71,
            )
        )
        _, report = apply_rows(rows, incident)
        assert report.funnel_violations == 2
        assert any("before injection" in note for note in report.notes)

    def test_a_cascade_bug_aborts_the_run_rather_than_returning_broken_events(self, monkeypatch):
        """Handing back a corrupt corpus with an honest report nobody reads is the failure this
        system argues against everywhere else. A cascade that stops halfway must abort."""
        from verdict import inject as module

        def careless(events, metric, chosen, **kwargs):
            filled = events.editable("is_filled")
            for i in chosen:
                filled[i] = 0

        monkeypatch.setattr(module, "_mutate", careless)
        rows = cell(200, country="AR", fill=1.0, render=1.0)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0,
                end=T0 + HOUR,
                magnitude=-0.5,
                seed=73,
            )
        )
        with pytest.raises(InjectionError, match="could not have happened"):
            apply_rows(rows, incident)

    def test_raising_render_rate_never_outruns_the_fills(self):
        rows = cell(1000, country="AR", fill=0.6, render=0.5)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="render_rate",
                where={"country": "AR"},
                start=T0,
                end=T0 + HOUR,
                magnitude=0.8,
                seed=4,
            )
        )
        out, _ = apply_rows(rows, incident)
        assert_funnel(out)
        assert sum(r["is_impression"] for r in out) <= sum(r["is_filled"] for r in out)


class TestShapesMoveWhatTheyClaim:
    def test_main_effect_moves_the_named_segment_only(self):
        rows, incident = scenario_main_effect()
        out, report = apply_rows(rows, incident)

        window = {"since": T0 + 6 * HOUR, "until": T0 + 18 * HOUR}
        target = shift(
            select(rows, {"country": "AR"}, **window),
            select(out, {"country": "AR"}, **window),
            "fill_rate",
        )
        assert target == pytest.approx(-0.35, abs=0.01)
        assert report.realised_magnitude == pytest.approx(target, abs=1e-9)

        for other in ("BR", "US", "IN"):
            assert select(rows, {"country": other}) == select(out, {"country": other})

    @pytest.mark.parametrize(
        ("metric", "magnitude"),
        [
            ("fill_rate", -0.30),
            ("fill_rate", 0.15),
            ("render_rate", -0.20),
            ("render_rate", 0.01),
            ("ctr", -0.45),
            ("ctr", 0.40),
            ("ecpm", -0.25),
            ("ecpm", 0.35),
        ],
    )
    def test_each_movable_metric_moves_by_the_requested_amount(self, metric, magnitude):
        rows = cell(20_000, span=6 * HOUR, country="AR", fill=0.78, render=0.90, ctr=0.05)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric=metric,
                where={"country": "AR"},
                start=T0,
                end=T0 + 6 * HOUR,
                magnitude=magnitude,
                seed=13,
            )
        )
        out, report = apply_rows(rows, incident)
        assert_funnel(out)
        assert shift(rows, out, metric) == pytest.approx(magnitude, abs=0.005)
        assert report.realised_magnitude == pytest.approx(magnitude, abs=0.005)

    @pytest.mark.parametrize("metric", ["fill_rate", "render_rate", "ctr", "ecpm"])
    def test_moving_one_metric_leaves_the_others_where_they_were(self, metric):
        """Collateral movement would be scored as a false positive against an incident the
        injector itself created, so it is a measurement bug rather than an inelegance.

        Lowering a rate stays clean because the rows are drawn uniformly and therefore carry
        the population's own downstream behaviour out with them. Raising one needs the
        promoted rows to be given that behaviour explicitly, which is the case this pins.
        """
        rows = cell(20_000, span=6 * HOUR, country="AR", fill=0.70, render=0.90, ctr=0.05)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric=metric,
                where={"country": "AR"},
                start=T0,
                end=T0 + 6 * HOUR,
                magnitude=0.20,
                seed=19,
            )
        )
        out, _ = apply_rows(rows, incident)
        for other in ("fill_rate", "render_rate", "ctr", "ecpm"):
            if other == metric:
                continue
            assert abs(shift(rows, out, other)) < 0.02, f"{metric} dragged {other} with it"

    def test_high_cardinality_hits_one_entity_and_is_flagged_as_a_blind_spot(self):
        rows, incident = scenario_high_cardinality()
        out, _ = apply_rows(rows, incident)

        moved = shift(
            select(rows, {"app_id": "app_03"}), select(out, {"app_id": "app_03"}), "fill_rate"
        )
        assert moved == pytest.approx(-0.30, abs=0.01)
        for i in range(6):
            if i == 3:
                continue
            app = f"app_{i:02d}"
            assert select(rows, {"app_id": app}) == select(out, {"app_id": app})

        assert incident.expected_detectable is False
        assert "lattice" in incident.blind_spot

    def test_slow_drift_ramps_to_the_target_rather_than_stepping_to_it(self):
        rows, incident = scenario_slow_drift()
        out, report = apply_rows(rows, incident)
        assert len(incident.edits) == 5

        by_day = [
            shift(
                select(rows, since=T0 + n * DAY, until=T0 + (n + 1) * DAY),
                select(out, since=T0 + n * DAY, until=T0 + (n + 1) * DAY),
                "fill_rate",
            )
            for n in range(5)
        ]
        assert by_day == sorted(by_day, reverse=True), "the ramp is not monotone"
        assert by_day[0] == pytest.approx(-0.05, abs=0.01)
        assert by_day[-1] == pytest.approx(-0.25, abs=0.01)

        # The headline magnitude is where the drift ended up, not the average of the ramp.
        # Scoring a drift against its mean would credit a detector for finding an incident half
        # the size of the one described.
        assert report.realised_magnitude == pytest.approx(by_day[-1], abs=1e-9)
        assert incident.expected_detectable is False

    def test_spike_is_confined_to_its_few_minutes(self):
        rows, incident = scenario_spike()
        out, report = apply_rows(rows, incident)
        start = T0 + 10 * HOUR
        end = start + timedelta(minutes=5)

        inside = shift(
            select(rows, since=start, until=end), select(out, since=start, until=end), "fill_rate"
        )
        assert inside == pytest.approx(-0.60, abs=0.03)
        assert select(rows, until=start) == select(out, until=start)
        assert select(rows, since=end) == select(out, since=end)

        # Diluted to near-invisibility by the hour it sits in, which is the whole point of the
        # shape: the finest stored grain is coarser than the incident.
        hour = {"since": T0 + 10 * HOUR, "until": T0 + 11 * HOUR}
        assert abs(shift(select(rows, **hour), select(out, **hour), "fill_rate")) < 0.06
        assert report.rows_changed > 0
        assert incident.expected_detectable is False


class TestInteractionStaysInvisible:
    """The subtle shape. Only the intersection moves, and it moves by an amount that each
    one-way marginal dilutes by its own cardinality. If either marginal moved on its own the
    incident would be findable without any structural reasoning, and a detector that only reads
    marginals would score a pass it has not earned."""

    def test_only_the_intersection_changes(self):
        rows, incident = scenario_interaction()
        out, _ = apply_rows(rows, incident)

        allowed = touched(rows, incident)
        for i, (was, now) in enumerate(zip(rows, out, strict=True)):
            if i not in allowed:
                assert was == now, f"row {i} is outside the cell and was modified"

        cell_shift = shift(
            select(rows, {"country": "C3", "device_model": "D5"}),
            select(out, {"country": "C3", "device_model": "D5"}),
            "fill_rate",
        )
        assert cell_shift == pytest.approx(-0.40, abs=0.01)

    def test_both_one_way_marginals_stay_flat(self):
        rows, incident = scenario_interaction()
        out, report = apply_rows(rows, incident)

        cell_rows = select(rows, {"country": "C3", "device_model": "D5"})
        cell_fills = sum(r["is_filled"] for r in cell_rows)
        cell_shift = shift(
            select(rows, {"country": "C3", "device_model": "D5"}),
            select(out, {"country": "C3", "device_model": "D5"}),
            "fill_rate",
        )

        for dimension, value in (("country", "C3"), ("device_model", "D5")):
            marginal_rows = select(rows, {dimension: value})
            marginal_shift = shift(marginal_rows, select(out, {dimension: value}), "fill_rate")

            # Small in absolute terms, and exactly as small as the arithmetic requires: the
            # marginal can only move by the cell's movement diluted by the cell's share of the
            # marginal's numerator. Asserting the identity rather than a hand-picked tolerance
            # is what makes this a guarantee instead of an observation about these numbers.
            share = cell_fills / sum(r["is_filled"] for r in marginal_rows)
            assert marginal_shift == pytest.approx(cell_shift * share, rel=1e-9)
            assert abs(marginal_shift) < 0.06
            assert report.marginal_shift[f"{dimension}={value}"] == pytest.approx(
                marginal_shift, rel=1e-9
            )

        # And the grand total, where it is diluted by both dimensions at once.
        assert abs(shift(rows, out, "fill_rate")) < 0.01

    def test_a_single_dimension_is_refused(self):
        with pytest.raises(InjectionError, match="intersection of two dimensions"):
            plan(
                IncidentSpec(
                    kind=KIND_INTERACTION,
                    metric="fill_rate",
                    where={"country": "C3"},
                    start=T0,
                    end=T0 + HOUR,
                    magnitude=-0.4,
                )
            )


class TestCompensatingPairCancels:
    """Two segments move hard in opposite directions and the grand total does not budge. A
    detector that only starts investigating once a top-line metric moves never opens a case
    here, which is exactly what this shape is planted to measure."""

    def test_the_grand_total_does_not_move_at_all(self):
        rows, incident = scenario_compensating_pair()
        out, report = apply_rows(rows, incident)

        # Integer equality, not a tolerance. The counterpart is sized in counted fills rather
        # than in percent, so the cancellation is exact by construction: two segments each
        # moving 30% of their own baseline would only cancel if they happened to be the same
        # size, and that would be an accident of the corpus rather than a property of the shape.
        assert sum(r["is_filled"] for r in out) == sum(r["is_filled"] for r in rows)
        assert len(out) == len(rows)
        assert shift(rows, out, "fill_rate") == 0.0
        assert report.marginal_shift["__all__"] == 0.0

    def test_both_segments_move_materially_and_in_opposite_directions(self):
        rows, incident = scenario_compensating_pair()
        out, _ = apply_rows(rows, incident)

        primary = shift(
            select(rows, {"country": "AR"}), select(out, {"country": "AR"}), "fill_rate"
        )
        counter = shift(
            select(rows, {"country": "BR"}), select(out, {"country": "BR"}), "fill_rate"
        )
        assert primary == pytest.approx(-0.30, abs=0.01)
        assert counter > 0.15
        assert select(rows, {"country": "US"}) == select(out, {"country": "US"})
        assert incident.counterpart_segment == {"country": "BR"}

    def test_a_price_pair_cancels_in_currency_rather_than_in_percent(self):
        """eCPM has no flag to flip, so its counterpart is sized by the revenue the primary
        gave up. Mirroring the percentage instead would leave a residue whenever the two
        segments earn different amounts, which is almost always."""
        rows = combine(
            cell(2000, span=6 * HOUR, country="AR"),
            cell(4000, span=6 * HOUR, country="BR"),
        )
        incident = plan(
            IncidentSpec(
                kind=KIND_COMPENSATING_PAIR,
                metric="ecpm",
                where={"country": "AR"},
                counterpart={"country": "BR"},
                start=T0,
                end=T0 + 6 * HOUR,
                magnitude=-0.25,
                seed=67,
            )
        )
        out, _ = apply_rows(rows, incident)

        assert shift(rows, out, "ecpm") == pytest.approx(0.0, abs=1e-12)
        primary = shift(select(rows, {"country": "AR"}), select(out, {"country": "AR"}), "ecpm")
        counter = shift(select(rows, {"country": "BR"}), select(out, {"country": "BR"}), "ecpm")
        assert primary == pytest.approx(-0.25, rel=1e-9)
        assert counter == pytest.approx(0.125, rel=0.01)
        # Only the price moved: no impression was created or destroyed on either side.
        assert sum(r["is_impression"] for r in out) == sum(r["is_impression"] for r in rows)

    def test_a_pair_without_a_counterpart_is_refused(self):
        with pytest.raises(InjectionError, match="counterpart"):
            plan(
                IncidentSpec(
                    kind=KIND_COMPENSATING_PAIR,
                    metric="fill_rate",
                    where={"country": "AR"},
                    start=T0,
                    end=T0 + HOUR,
                    magnitude=-0.3,
                )
            )


class TestCleanControl:
    """Without a run that plants nothing, the false positive rate is unmeasurable and a system
    that invents three confident incidents on untouched data looks like one that works."""

    def test_nothing_changes_row_for_row(self):
        rows, incident = scenario_clean()
        out, report = apply_rows(rows, incident)

        assert out == rows
        assert report.rows_changed == 0
        assert report.rows_matched == 0
        assert report.realised_magnitude == 0.0
        assert report.funnel_violations == 0

    def test_the_arrow_table_comes_back_untouched(self):
        rows, incident = scenario_clean()
        table = to_arrow(rows)
        out, report = apply(table, incident)

        assert out.equals(table)
        assert out.schema.equals(table.schema)
        assert report.rows_changed == 0

    def test_the_answer_key_says_the_correct_output_is_nothing(self):
        _, incident = scenario_clean()
        assert incident.expected_direction == "none"
        assert incident.expected_segment == {}
        assert incident.expected_detectable is False
        assert "false positive" in incident.description

    def test_a_control_with_a_magnitude_is_refused(self):
        with pytest.raises(InjectionError, match="control exists to change nothing"):
            plan(
                IncidentSpec(
                    kind=KIND_CLEAN,
                    metric="fill_rate",
                    where={},
                    start=T0,
                    end=T0 + HOUR,
                    magnitude=-0.3,
                )
            )


class TestDeterminism:
    def test_the_same_seed_reproduces_the_run_exactly(self):
        rows, incident = scenario_main_effect()
        first, report_a = apply_rows(rows, incident)
        second, report_b = apply_rows(rows, incident)
        assert first == second
        assert report_a == report_b

    def test_the_same_seed_reproduces_an_arrow_run_exactly(self):
        rows, incident = scenario_main_effect()
        table = to_arrow(rows)
        first, _ = apply(table, incident)
        second, _ = apply(table, incident)
        assert first.equals(second)

    def test_a_different_seed_selects_different_rows(self):
        rows, _ = scenario_main_effect()

        def run(seed: int) -> list[dict]:
            incident = plan(
                IncidentSpec(
                    kind=KIND_MAIN_EFFECT,
                    metric="fill_rate",
                    where={"country": "AR"},
                    start=T0 + 6 * HOUR,
                    end=T0 + 18 * HOUR,
                    magnitude=-0.35,
                    seed=seed,
                )
            )
            return apply_rows(rows, incident)[0]

        assert run(11) != run(12)
        # Same incident either way, though: only which rows carry it differs.
        assert measure(run(11), "fill_rate") == pytest.approx(measure(run(12), "fill_rate"))

    def test_the_caller_s_rows_are_never_mutated(self):
        """A harness that injects three variants from one loaded corpus would otherwise be
        comparing each run against an input the previous run had already edited."""
        rows, incident = scenario_main_effect()
        snapshot = [dict(r) for r in rows]
        apply_rows(rows, incident)
        assert rows == snapshot


class TestHonestReporting:
    def test_realised_matches_requested_on_a_large_population(self):
        rows = cell(50_000, span=6 * HOUR, country="AR")
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0,
                end=T0 + 6 * HOUR,
                magnitude=-0.35,
                seed=31,
            )
        )
        _, report = apply_rows(rows, incident)
        assert report.realised_magnitude == pytest.approx(-0.35, abs=0.001)
        assert report.rows_examined == 50_000
        assert report.rows_matched == 50_000

    def test_realised_records_what_rounding_actually_delivered(self):
        """Asking for 35% off five fills cannot deliver 35%. Recording the request rather than
        the delivery would measure localization accuracy against a number that was never true
        of the data."""
        rows = cell(10, span=HOUR, country="AR", fill=0.5, render=1.0, ctr=0.0)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0,
                end=T0 + HOUR,
                magnitude=-0.35,
                seed=37,
            )
        )
        out, report = apply_rows(rows, incident)
        assert sum(r["is_filled"] for r in out) == 3
        assert report.realised_magnitude == pytest.approx(-0.40)
        assert report.realised_magnitude != incident.spec.magnitude

    def test_an_undeliverable_rise_is_reported_rather_than_silently_shrunk(self):
        """Raising a rate is bounded by how many rows have room to move. Ninety-nine of a
        hundred requests already filled leaves exactly one, and under-delivering by a factor of
        twenty without saying so would corrupt every accuracy number computed from the run."""
        rows = cell(100, span=HOUR, country="AR", fill=0.99)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0,
                end=T0 + HOUR,
                magnitude=0.20,
                seed=41,
            )
        )
        out, report = apply_rows(rows, incident)

        assert report.rows_eligible == 1
        assert report.rows_changed == 1
        assert report.realised_magnitude == pytest.approx(1 / 99, abs=1e-6)
        assert report.realised_magnitude < 0.20
        assert any("only 1" in note for note in report.notes)
        assert sum(r["is_filled"] for r in out) == 100

    def test_a_segment_with_no_traffic_says_so_instead_of_dividing_by_zero(self):
        rows = cell(100, span=HOUR, country="AR")
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="ctr",
                where={"country": "ZZ"},
                start=T0,
                end=T0 + HOUR,
                magnitude=-0.5,
                seed=43,
            )
        )
        out, report = apply_rows(rows, incident)
        assert out == rows
        assert report.rows_matched == 0
        assert report.realised_magnitude == 0.0
        assert any("nothing to move" in note for note in report.notes)


class TestBoundaries:
    @pytest.mark.parametrize("kind", KINDS)
    def test_no_shape_touches_a_row_outside_its_segment_and_window(self, kind):
        rows, incident = SCENARIOS[kind]()
        out, _ = apply_rows(rows, incident)
        allowed = touched(rows, incident)
        for i, (was, now) in enumerate(zip(rows, out, strict=True)):
            if i not in allowed:
                assert was == now, f"{kind} modified row {i}, which is out of scope"

    def test_the_window_is_half_open_at_both_ends(self):
        """Adjacent windows must not both claim the same row, or a drift ramp would inject
        twice into every boundary instant and the final magnitude would overshoot."""
        rows = cell(240, span=4 * HOUR, country="AR", fill=1.0, render=0.0, ctr=0.0)
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0 + HOUR,
                end=T0 + 2 * HOUR,
                magnitude=-1.0 + 1e-9,
                seed=47,
            )
        )
        out, _ = apply_rows(rows, incident)
        changed = [r["event_time"] for was, r in zip(rows, out, strict=True) if was != r]
        assert min(changed) >= T0 + HOUR
        assert max(changed) < T0 + 2 * HOUR

    def test_a_timezone_aware_spec_lines_up_with_naive_event_times(self):
        """The corpus stores naive timestamps that the DDL declares UTC. Comparing those
        against an aware bound would raise, and reconciling them by assuming local time would
        move the window hours away from the incident it describes."""
        rows = cell(600, span=6 * HOUR, country="AR")
        aware = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=(T0 + HOUR).replace(tzinfo=timezone.utc),
                end=(T0 + 3 * HOUR).replace(tzinfo=timezone.utc),
                magnitude=-0.5,
                seed=53,
            )
        )
        naive = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"country": "AR"},
                start=T0 + HOUR,
                end=T0 + 3 * HOUR,
                magnitude=-0.5,
                seed=53,
            )
        )
        assert apply_rows(rows, aware)[0] == apply_rows(rows, naive)[0]


class TestArrowAndRowPathsAgree:
    """Two implementations of the funnel cascade would be two chances to get it wrong, and the
    difference would only show up in whichever path the tests did not exercise."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_both_entry_points_produce_the_same_events(self, kind):
        rows, incident = SCENARIOS[kind]()
        from_rows, row_report = apply_rows(rows, incident)
        table, arrow_report = apply(to_arrow(rows), incident)

        expected = [{k: r[k] for k in COLUMNS} for r in from_rows]
        assert table.to_pylist() == expected
        assert row_report == arrow_report

    def test_column_types_survive_the_round_trip(self):
        """A uint8 flag widened to int64 on the way through would be rejected or silently
        coerced at insert time, long after this reported success."""
        rows, incident = scenario_main_effect()
        table = to_arrow(rows)
        out, _ = apply(table, incident)
        assert out.schema.equals(table.schema)

    def test_untouched_columns_are_handed_back_as_they_arrived(self):
        rows, incident = scenario_main_effect()
        table = to_arrow(rows)
        out, _ = apply(table, incident)
        for name in ("event_time", "app_id", "country", "device_model", "geo_device_id"):
            assert out.column(name).equals(table.column(name))

    def test_a_table_missing_a_funnel_column_is_refused(self):
        rows, incident = scenario_main_effect()
        table = to_arrow(rows).drop_columns(["revenue"])
        with pytest.raises(InjectionError, match="revenue"):
            apply(table, incident)


class TestDimensionResolution:
    """Seven of the nine lattice dimensions are dictionary lookups rather than columns on
    ad_events, so an injector that could only match physical columns could not plant an
    incident on country, os_version or publisher_tier at all."""

    def test_a_dimension_absent_from_the_fact_table_is_resolved_through_the_index(self):
        rows = combine(
            cell(1500, span=6 * HOUR, geo_device_id="gd_a"),
            cell(1500, span=6 * HOUR, geo_device_id="gd_b"),
        )
        index = DimensionIndex(
            {"os_version": ("geo_device_id", {"gd_a": "Android 15", "gd_b": "Android 14"})}
        )
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"os_version": "Android 15"},
                start=T0,
                end=T0 + 6 * HOUR,
                magnitude=-0.30,
                seed=59,
            )
        )
        out, report = apply_rows(rows, incident, dimensions=index)

        assert report.rows_matched == 1500
        moved = shift(
            select(rows, {"geo_device_id": "gd_a"}),
            select(out, {"geo_device_id": "gd_a"}),
            "fill_rate",
        )
        assert moved == pytest.approx(-0.30, abs=0.01)
        assert select(rows, {"geo_device_id": "gd_b"}) == select(out, {"geo_device_id": "gd_b"})

    def test_an_unresolvable_dimension_fails_loudly(self):
        """Matching nothing would report a rows_changed of zero, which reads exactly like a
        clean control and would be scored as one."""
        rows = cell(100, span=HOUR, country="AR")
        incident = plan(
            IncidentSpec(
                kind=KIND_MAIN_EFFECT,
                metric="fill_rate",
                where={"os_versoin": "Android 15"},
                start=T0,
                end=T0 + HOUR,
                magnitude=-0.3,
                seed=61,
            )
        )
        with pytest.raises(InjectionError, match="Cannot resolve dimension"):
            apply_rows(rows, incident)

    def test_an_index_reads_a_dimension_csv(self, tmp_path: Path):
        path = tmp_path / "geo_device.csv"
        path.write_text(
            "geo_device_id,region,country,device_model,os_version\n"
            "gd_a,APAC,IN,Galaxy S23,Android 15\n"
            "gd_b,EU,DE,Pixel 8,Android 14\n"
        )
        index = DimensionIndex.from_csv(path, "geo_device_id")
        assert index.dimensions == ["country", "device_model", "os_version", "region"]
        assert index.lookups["country"][1]["gd_a"] == "IN"

    def test_an_index_with_the_wrong_key_column_is_refused(self, tmp_path: Path):
        path = tmp_path / "apps.csv"
        path.write_text("app_id,category\napp_00,Games\n")
        with pytest.raises(InjectionError, match="no column 'geo_device_id'"):
            DimensionIndex.from_csv(path, "geo_device_id")


class TestPlanValidation:
    """Every one of these is a spec that would otherwise run to completion and write an answer
    key describing something the data does not contain."""

    def _spec(self, **overrides):
        base = dict(
            kind=KIND_MAIN_EFFECT,
            metric="fill_rate",
            where={"country": "AR"},
            start=T0,
            end=T0 + HOUR,
            magnitude=-0.3,
        )
        base.update(overrides)
        return IncidentSpec(**base)

    def test_unknown_kind(self):
        with pytest.raises(InjectionError, match="Unknown kind"):
            plan(self._spec(kind="gradual_collapse"))

    def test_unknown_metric(self):
        with pytest.raises(InjectionError, match="Cannot move"):
            plan(self._spec(metric="requests"))

    def test_inverted_window(self):
        with pytest.raises(InjectionError, match="not before"):
            plan(self._spec(start=T0 + HOUR, end=T0))

    def test_zero_magnitude(self):
        with pytest.raises(InjectionError, match="kind='clean'"):
            plan(self._spec(magnitude=0.0))

    def test_magnitude_past_total_collapse(self):
        with pytest.raises(InjectionError, match="bounded at zero"):
            plan(self._spec(magnitude=-1.0))

    def test_empty_segment(self):
        with pytest.raises(InjectionError, match="needs a segment"):
            plan(self._spec(where={}))

    def test_fill_rate_cannot_be_sliced_by_a_post_fill_dimension(self):
        """The trap that makes this worth checking: clearing a fill removes the row from an
        advertiser-keyed segment altogether, so the segment's fill rate stays pinned at 1.0
        and the answer key records a drop no query can reproduce."""
        for dimension in ("advertiser_id", "vertical", "campaign_type"):
            with pytest.raises(InjectionError, match="only exists"):
                plan(self._spec(where={dimension: "adv_1"}))

    def test_post_fill_dimensions_are_fine_for_metrics_measured_over_impressions(self):
        incident = plan(self._spec(metric="ctr", where={"vertical": "Retail"}))
        assert incident.expected_segment == {"vertical": "Retail"}

    def test_high_cardinality_must_name_a_key_outside_the_lattice(self):
        with pytest.raises(InjectionError, match="high_cardinality must target"):
            plan(self._spec(kind=KIND_HIGH_CARDINALITY, where={"country": "AR"}))

    def test_a_spike_may_not_last_an_hour(self):
        with pytest.raises(InjectionError, match="grain sensitivity"):
            plan(self._spec(kind=KIND_SPIKE, end=T0 + 2 * HOUR))

    def test_a_pair_may_not_compensate_against_itself(self):
        with pytest.raises(InjectionError, match="same segment"):
            plan(
                self._spec(
                    kind=KIND_COMPENSATING_PAIR,
                    where={"country": "AR"},
                    counterpart={"country": "AR"},
                )
            )

    def test_a_drift_needs_more_than_one_step(self):
        with pytest.raises(InjectionError, match="at least two steps"):
            plan(self._spec(kind=KIND_SLOW_DRIFT, steps=1))


class TestAnswerKey:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_key_survives_json_dumps_with_no_custom_encoder(self, kind):
        _, incident = SCENARIOS[kind]()
        encoded = json.dumps(incident.to_dict())
        restored = json.loads(encoded)
        assert restored["kind"] == kind
        assert restored["description"] == incident.description
        assert isinstance(restored["start"], str)

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_key_records_the_outcome_only_after_it_has_one(self, kind):
        rows, incident = SCENARIOS[kind]()
        assert incident.applied is False
        assert incident.realised_magnitude is None

        _, report = apply_rows(rows, incident)
        final = incident.with_outcome(report)
        assert final.applied is True
        assert final.rows_affected == report.rows_changed
        assert final.realised_magnitude == report.realised_magnitude
        assert json.dumps(final.to_dict())

    def test_an_unapplied_key_is_marked_so_it_cannot_be_scored_by_accident(self):
        """A key written from a plan that never ran carries a rows_affected of zero and a null
        realised magnitude, and scoring against it marks every injected incident as a miss."""
        _, incident = scenario_main_effect()
        assert incident.to_dict()["applied"] is False
        assert incident.to_dict()["realised_magnitude"] is None

    def test_direction_and_segment_are_the_ground_truth_a_verdict_is_scored_against(self):
        _, incident = scenario_main_effect()
        assert incident.expected_segment == {"country": "AR"}
        assert incident.expected_metric == "fill_rate"
        assert incident.expected_direction == "fall"
        assert incident.expected_detectable is True
        assert incident.blind_spot == ""

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_description_reads_as_a_results_table_entry(self, kind):
        _, incident = SCENARIOS[kind]()
        text = incident.description
        assert text.endswith(".")
        assert "\u2014" not in text, "unicode em-dash; the house style writes ' -- '"
        if kind == KIND_CLEAN:
            assert "no incident is injected" in text
            return
        assert incident.expected_metric in text
        for value in incident.expected_segment.values():
            assert value in text

    def test_shapes_expected_to_be_missed_say_why(self):
        for kind in (KIND_HIGH_CARDINALITY, KIND_SLOW_DRIFT, KIND_SPIKE, KIND_CLEAN):
            _, incident = SCENARIOS[kind]()
            assert incident.expected_detectable is False
            assert incident.blind_spot
        for kind in (KIND_MAIN_EFFECT, KIND_INTERACTION, KIND_COMPENSATING_PAIR):
            _, incident = SCENARIOS[kind]()
            assert incident.expected_detectable is True


class TestFormulasAgreeWithTheRegistry:
    """The injector restates the four formulas it can move rather than importing them.

    If the answer key and the detector both divided by the same wrong denominator, the injected
    incident and the resulting finding would agree and the evaluation would score a broken
    system as correct. Keeping two independent statements only helps if a test compares them,
    which is this one.
    """

    def test_numerators_denominators_and_scales_match_metrics_yaml(self):
        from verdict.inject import _FORMULA
        from verdict.metrics import MetricRegistry

        path = Path(__file__).resolve().parents[1] / "config" / "metrics.yaml"
        registry = MetricRegistry.load(path)
        for name, (numerator, denominator, scale) in _FORMULA.items():
            metric = registry.metric(name)
            assert metric.numerator_field == numerator, name
            assert metric.denominator_field == denominator, name
            assert metric.scale == scale, name
