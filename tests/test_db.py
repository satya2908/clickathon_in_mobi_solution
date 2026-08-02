"""Parameter binding, which is where a whole class of silent wrongness lives.

The timezone case is here because it cost real correctness and left no trace. Every bucket
column is DateTime('UTC'); the driver reads a naive datetime as local time and converts it on
the way out. On a machine at +05:30 a window asked for as midnight reached the server as 18:30
the previous day, and because the baselines shifted by the same amount the findings still
looked entirely reasonable. What gave it away was a query printed next to its own answer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from verdict.db import as_utc, render_sql, rows_as_utc


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

    def test_datetimes_inside_an_array_parameter_are_stamped_too(self):
        """An Array(DateTime) bound element by element must not smuggle local time through.

        Left unstamped, a query filtering on such an array silently matches fewer rows than it
        should and reports no error, which is how a batched recurrence lookup came back empty.
        """
        out = as_utc({"cutoffs": [datetime(2026, 6, 23), datetime(2026, 6, 24, 9, 15)]})
        assert out["cutoffs"] == [
            datetime(2026, 6, 23, tzinfo=UTC),
            datetime(2026, 6, 24, 9, 15, tzinfo=UTC),
        ]

    def test_arrays_of_other_things_are_unharmed(self):
        assert as_utc({"fps": ["a", "b"], "ks": [1, 2]}) == {"fps": ["a", "b"], "ks": [1, 2]}


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


class TestWritingIsAlsoNormalised:
    """Reading was only half of it. A naive datetime handed to insert is shifted by the host
    offset exactly as a query parameter was, so a case investigated at midnight was stored as
    18:30 the previous day -- on a half-hour boundary no hourly bucket can match."""

    def test_a_naive_row_value_is_read_as_utc(self):
        [[stamped]] = rows_as_utc([[datetime(2026, 6, 23, 0, 0)]])
        assert stamped == datetime(2026, 6, 23, 0, 0, tzinfo=UTC)

    def test_an_aware_value_is_left_where_it_was(self):
        aware = datetime(2026, 6, 23, 0, 0, tzinfo=UTC)
        [[stamped]] = rows_as_utc([[aware]])
        assert stamped is aware

    def test_everything_else_passes_through(self):
        [row] = rows_as_utc([["fill_rate", 3, 0.44, None]])
        assert row == ["fill_rate", 3, 0.44, None]

    def test_the_window_survives_a_round_trip_through_a_row(self):
        """The shape the bug actually took: midnight in, midnight out, on any host."""
        start = datetime(2026, 6, 23, 0, 0)
        [[stamped]] = rows_as_utc([[start]])
        assert stamped.utcoffset() == timedelta(0)
        assert stamped.hour == 0 and stamped.minute == 0
