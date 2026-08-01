"""The tracer, and specifically the identifier a case carries away from it.

A case stores one id so a reader can open the same investigation in HyperDX. Getting it wrong is
invisible from inside the system: the field is populated, the insert succeeds, the case file
renders, and the only symptom is that a trace viewer handed an unknown id shows an empty result
rather than an error. It stored a span id -- 16 hex characters where a trace id is 32 -- so every
stored value matched a SpanId in otel_traces and none matched a TraceId.

Widths are asserted directly because that is the property that failed, and it is the one thing a
mistake here cannot fake.
"""

from __future__ import annotations

import time

from verdict.config import TracingConfig
from verdict.trace import Tracer

TRACE_ID_HEX = 32
SPAN_ID_HEX = 16


def tracer() -> Tracer:
    # Exporting is off; the SDK still assigns real ids, which is all these tests read.
    return Tracer(TracingConfig(enabled=True, endpoint=""))


class TestTheCaseCarriesATraceIdNotASpanId:
    def test_trace_id_is_thirty_two_hex_characters(self):
        t = tracer()
        with t.span("detect"):
            pass
        assert len(t.trace_id) == TRACE_ID_HEX
        assert int(t.trace_id, 16) > 0

    def test_span_id_is_sixteen_and_they_are_not_the_same_value(self):
        t = tracer()
        with t.span("detect"):
            pass
        step = t.steps[0]
        assert len(step.span_id) == SPAN_ID_HEX
        assert step.trace_id != step.span_id

    def test_consecutive_top_level_spans_start_separate_traces(self):
        """Not a quirk to work around -- it is why the pipeline needs a root span.

        Each top-level span begins a new trace, so a run whose stages were all top-level
        fragmented into one trace per stage: 56 traces across 287 spans, measured. The deep link
        on a case then opened one fragment. `investigate` wraps the run for this reason.
        """
        t = tracer()
        with t.span("detect"):
            pass
        with t.span("localize"):
            pass
        assert len({s.trace_id for s in t.steps}) == 2

    def test_a_root_span_gathers_the_stages_into_one_trace(self):
        t = tracer()
        with t.span("investigate"):
            with t.span("detect"):
                pass
            with t.span("localize"):
                pass
        assert len({s.trace_id for s in t.steps}) == 1

    def test_nested_spans_share_the_trace_but_not_the_span(self):
        t = tracer()
        with t.span("detect"):
            with t.span("temporal:fill_rate"):
                pass
        outer, inner = t.steps[0], t.steps[1]
        assert outer.trace_id == inner.trace_id
        assert outer.span_id != inner.span_id

    def test_no_spans_means_no_id_rather_than_a_fabricated_one(self):
        assert tracer().trace_id == ""


class TestStepRowsMatchTheStoredColumns:
    def test_trace_id_is_not_written_into_case_steps(self):
        """case_steps has no such column; the trace belongs to the case, not to each step."""
        t = tracer()
        with t.span("detect"):
            pass
        assert "trace_id" not in t.steps[0].as_row()

    def test_the_row_still_carries_the_span_id(self):
        t = tracer()
        with t.span("detect"):
            pass
        assert t.steps[0].as_row()["span_id"] == t.steps[0].span_id


class TestAnOpenSpanReportsElapsedRatherThanZero:
    """Cases are built inside the run's root span, so the root is always still open when its row
    is taken. Reporting 0 put a zero-width bar on the one step that sets a waterfall's scale."""

    def test_a_closed_span_reports_its_real_duration(self):
        t = tracer()
        with t.span("detect"):
            time.sleep(0.02)
        assert t.steps[0].as_row()["duration_ms"] >= 15

    def test_an_open_span_reports_time_so_far(self):
        t = tracer()
        with t.span("investigate"):
            time.sleep(0.02)
            row = t.steps[0].as_row()
        assert row["duration_ms"] >= 15

    def test_the_recorded_duration_wins_once_the_span_closes(self):
        """Elapsed-so-far is a stand-in, not a replacement: it must not drift afterwards."""
        t = tracer()
        with t.span("detect"):
            pass
        t.steps[0].duration_ms = 4242
        assert t.steps[0].as_row()["duration_ms"] == 4242

    def test_a_span_that_finished_instantly_still_reports_zero(self):
        """A closed sub-millisecond step and an open one both hold 0, and conflating them made
        fast children report longer than the parents they ran inside."""
        t = tracer()
        with t.span("investigate"):
            with t.span("fast"):
                pass
            time.sleep(0.03)
            rows = [s.as_row() for s in t.steps]
        assert rows[1]["duration_ms"] == 0

    def test_no_child_outlasts_its_parent(self):
        t = tracer()
        with t.span("investigate"):
            with t.span("detect"):
                with t.span("temporal:fill_rate"):
                    pass
            time.sleep(0.03)
            rows = {s.name: s.as_row()["duration_ms"] for s in t.steps}
        assert rows["temporal:fill_rate"] <= rows["detect"] <= rows["investigate"]
