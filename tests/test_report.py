"""The case file is a claim about the system, so its claims have to hold.

The reproduction query is the part worth testing hardest. A page that shows a query alongside
a number is asserting that running the query yields the number, and an assertion nobody checks
is worse than no assertion at all -- a reader who runs a broken query concludes the *system* is
broken, not the report. The first version shipped here failed on ClickHouse with
ILLEGAL_AGGREGATION because `sum(fills) AS fills` resolves the alias inside its own aggregate,
and every count metric is named after the column its expression sums.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from verdict.metrics import MetricRegistry
from verdict.query import COUNTER_COLUMNS, Segment, Window
from verdict.report import _alias, _metric_tree, build_payload, reproduction_sql

REGISTRY = MetricRegistry.load("config/metrics.yaml")
WINDOW = Window(start=datetime(2026, 6, 23), end=datetime(2026, 6, 26), grain="1h")


# Stand-ins rather than real Cases. The report reads a narrow surface of each object, and
# constructing genuine ones would drag in the detector, the localizer and a database
# connection to test a string builder. What matters is that the shapes match, which the
# attribute names enforce: a rename upstream breaks these immediately.
class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def case_factory():
    def make(metric: str = "fill_rate", segment: Segment | None = None, accused: bool = True):
        seg = segment or Segment((("os_version", "Android 15"),))
        test = _Stub(
            observed=0.4526, expected=0.7855, z=-42.1, p_value=1.2e-14,
            model="two_proportion", direction="fall", relative_effect=-0.424,
        )
        finding = _Stub(
            metric=metric, segment=seg, window=WINDOW, detector="temporal", test=test,
            phi=1.87, weeks_kept=3, weeks_seen=4, survives_correction=True,
            effect_threshold=0.05, resolvable_effect=0.012, relative_effect=-0.424,
        )
        candidate = _Stub(
            segment=seg, relative_effect=-0.424, observed_value=0.4526, expected_value=0.7855,
            checks={}, status="accused", reason="", predicted_if_innocent=None,
        )
        localization = _Stub(
            metric=metric, window=WINDOW, parent=Segment.total(),
            parent_observed=0.715, parent_expected=0.785, parent_deviation=-0.0703,
            accused=candidate if accused else None,
            candidates=[candidate], cleared=[], mode="explain_away", note="",
        )
        return _Stub(
            case_id="c1", finding=finding, localization=localization,
            impact=_Stub(revenue=65.0, direct=False), confidence_value=0.92,
            confidence_json="{}", narrative="", narrative_source="template",
            recurrence_of="", segment=seg, verdict_kind="localized" if accused else "detected",
            series=[],
        )

    return make


@pytest.fixture
def result_factory(case_factory):
    def make(cases: int = 1, accused: bool = True, metric: str = "fill_rate"):
        return _Stub(
            run_id="r1", window=WINDOW, cells_tested=1063, findings_after_correction=310,
            metrics_scanned=[metric], persisted=False,
            cases=[case_factory(metric=metric, accused=accused) for _ in range(cases)],
            gaps=[_Stub(metric=metric, reason="below_detection_floor") for _ in range(168)],
        )

    return make


class TestAliasesCannotCollideWithColumns:
    @pytest.mark.parametrize("column", COUNTER_COLUMNS)
    def test_a_metric_named_after_a_column_is_renamed(self, column):
        assert _alias(column) != column

    @pytest.mark.parametrize("name", ["fill_rate", "render_rate", "ctr", "ecpm", "rpr"])
    def test_a_metric_not_named_after_a_column_keeps_its_name(self, name):
        assert _alias(name) == name

    def test_no_rendered_alias_equals_a_counter_column(self, case_factory):
        """The whole class of bug, caught statically rather than at the point a reader pastes."""
        for name in REGISTRY.metrics:
            expr = REGISTRY.metric(name).value_sql(from_rollup=True)
            sql = reproduction_sql(case_factory(metric=name), "rollup_1h", expr)
            aliases = {
                part.split(" AS ")[-1].strip().rstrip(",")
                for line in sql.splitlines()
                for part in line.split(",")
                if " AS " in part
            }
            assert not (aliases & set(COUNTER_COLUMNS)), f"{name} aliases collide: {aliases}"


class TestReproductionQuery:
    def test_it_selects_the_segment_the_case_accuses(self, case_factory):
        case = case_factory(segment=Segment((("os_version", "Android 15"),)))
        sql = reproduction_sql(case, "rollup_1h", "sum(fills) / nullIf(sum(requests), 0)")
        assert "combo = 'os_version'" in sql
        assert "key_a = 'Android 15'" in sql

    def test_a_quote_in_a_segment_value_is_escaped(self, case_factory):
        case = case_factory(segment=Segment((("app_id", "it's"),)))
        assert "it\\'s" in reproduction_sql(case, "rollup_1h", "sum(revenue)")

    def test_the_shown_expression_is_the_one_the_detector_used(self, result_factory):
        """The only guarantee that makes the panel worth showing.

        A query that runs, returns a plausible number and quietly disagrees with the verdict
        beside it is worse than no query at all: a reader who checks concludes the system is
        broken rather than the report.
        """
        payload = build_payload(result_factory(cases=1, accused=True), REGISTRY)
        expected = REGISTRY.metric("fill_rate").value_sql(from_rollup=True)
        assert expected in payload["cases"][0]["sql"]

    def test_an_unresolvable_metric_shows_nothing_rather_than_a_guess(self, result_factory):
        payload = build_payload(result_factory(cases=1, accused=True), registry=None)
        assert payload["cases"][0]["sql"] == ""


class TestMetricTreeDoesNotPaintUntestedAsClean:
    def test_a_metric_with_a_localized_case_is_red(self, result_factory):
        tree = _metric_tree(result_factory(cases=1, accused=True))
        assert tree[0]["state"] == "bad"

    def test_a_metric_detected_but_unattributed_is_amber_not_red(self, result_factory):
        tree = _metric_tree(result_factory(cases=1, accused=False))
        assert tree[0]["state"] == "warn"


class TestPayloadIsSelfContained:
    def test_every_case_carries_its_own_provenance(self, result_factory):
        payload = build_payload(result_factory(cases=1, accused=True), REGISTRY)
        assert payload["cases"][0]["sql"].strip()

    def test_recorded_query_shapes_are_carried_through(self, result_factory):
        payload = build_payload(
            result_factory(cases=0, accused=False), REGISTRY, queries={"rollup_slice": "SELECT 1"}
        )
        assert payload["queries"] == [{"name": "rollup_slice", "sql": "SELECT 1"}]
