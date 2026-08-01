"""Parameter binding, which is where a whole class of silent wrongness lives.

The timezone case is here because it cost real correctness and left no trace. Every bucket
column is DateTime('UTC'); the driver reads a naive datetime as local time and converts it on
the way out. On a machine at +05:30 a window asked for as midnight reached the server as 18:30
the previous day, and because the baselines shifted by the same amount the findings still
looked entirely reasonable. What gave it away was a query printed next to its own answer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from verdict.db import as_utc, render_sql


class TestNaiveDatetimesAreReadAsUTC:
    def test_a_naive_datetime_is_stamped_utc(self):
        out = as_utc({"s": datetime(2026, 6, 23)})
        assert out["s"] == datetime(2026, 6, 23, tzinfo=UTC)

    def test_stamping_does_not_shift_the_wall_clock(self):
        """The point is to declare the clock, not to convert between clocks."""
        out = as_utc({"s": datetime(2026, 6, 23, 14, 30)})
        assert (out["s"].hour, out["s"].minute) == (14, 30)

    def test_an_already_aware_datetime_is_left_alone(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        given = datetime(2026, 6, 23, tzinfo=ist)
        assert as_utc({"s": given})["s"] is given

    def test_non_datetime_parameters_pass_through(self):
        assert as_utc({"combo": "os_version", "k": 4}) == {"combo": "os_version", "k": 4}

    def test_empty_and_none_survive(self):
        assert as_utc(None) is None
        assert as_utc({}) == {}


class TestRenderedSQLMatchesWhatRan:
    """The displayed query is a claim that running it reproduces the number beside it."""

    def test_a_datetime_renders_as_a_quoted_literal(self):
        sql = render_sql("WHERE b >= {s:DateTime}", {"s": datetime(2026, 6, 23)})
        assert sql == "WHERE b >= '2026-06-23 00:00:00'"

    def test_a_string_is_quoted_and_escaped(self):
        sql = render_sql("WHERE k = {a:String}", {"a": "it's"})
        assert sql == "WHERE k = 'it\\'s'"

    def test_every_occurrence_of_a_placeholder_is_replaced(self):
        sql = render_sql("{s:DateTime} .. {s:DateTime}", {"s": datetime(2026, 1, 1)})
        assert "{s:" not in sql

    def test_a_placeholder_with_no_parameter_is_left_visible(self):
        """Better an obviously unrendered query than a plausible one missing a filter."""
        assert "{e:DateTime}" in render_sql("WHERE b < {e:DateTime}", {})
