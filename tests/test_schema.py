"""Tests for generated DDL and the grain/depth policy.

The failure this file guards against is silent. If storage writes a shallower lattice than the
readers ask for, every query still succeeds -- it just returns nothing, and a combo with no rows
is indistinguishable from a combo whose segments genuinely had no traffic. The detector would
report full coverage of a lattice it never actually read.
"""

from pathlib import Path

import pytest

from verdict.config import load_config
from verdict.detect import lattice_combos
from verdict.metrics import MetricRegistry
from verdict.schema import (
    GRAINS,
    LATTICE_DEPTH,
    ROLLUP_SOURCE,
    TOTAL_COMBO,
    all_statements,
    backfill_statements,
    cascaded_grains,
    combos,
    view_ddl,
    view_name,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = MetricRegistry.load(ROOT / "config" / "metrics.yaml")
DIMS = REGISTRY.lattice_dimensions


class TestLatticeDepth:
    def test_every_grain_declares_a_depth(self):
        assert set(LATTICE_DEPTH) == set(GRAINS)

    def test_depth_one_stops_at_single_dimensions(self):
        shallow = combos(DIMS, max_depth=1)
        assert len(shallow) == 1 + len(DIMS)
        assert all(b is None for _, b in shallow)

    def test_depth_two_adds_every_unordered_pair(self):
        n = len(DIMS)
        assert len(combos(DIMS, max_depth=2)) == 1 + n + n * (n - 1) // 2

    def test_the_grand_total_survives_at_every_depth(self):
        """Whatever else is cut, the parent every sufficiency test compares against remains."""
        for depth in (0, 1, 2):
            assert combos(DIMS, max_depth=depth)[0] == (TOTAL_COMBO, None)

    def test_the_five_minute_tier_is_shallow(self):
        """Not a style choice. A two-way cell at 5-minute grain holds a measured median of one
        request against a requirement of roughly 454, so those 1647 combos are 96% of the rows
        and zero of the testable cells."""
        assert LATTICE_DEPTH["5m"] == 1
        assert LATTICE_DEPTH["1h"] == 2
        assert LATTICE_DEPTH["1d"] == 2


class TestReadersMatchStorage:
    """The invariant: no reader asks a grain for cells that grain does not store."""

    @pytest.mark.parametrize("grain", sorted(GRAINS))
    @pytest.mark.parametrize("metric_name", ["fill_rate", "ctr", "revenue", "ecpm"])
    def test_detector_never_requests_a_combo_deeper_than_the_grain(self, grain, metric_name):
        metric = REGISTRY.metric(metric_name)
        depth = LATTICE_DEPTH[grain]
        for combo in lattice_combos(REGISTRY, metric, grain):
            if combo == TOTAL_COMBO:
                continue
            assert len(combo.split("|")) <= depth, (
                f"{grain} stores depth {depth} but the detector asked for {combo!r}"
            )

    def test_the_detector_still_reaches_two_way_cells_somewhere(self):
        """Guard against 'fixing' the mismatch by making every grain shallow."""
        metric = REGISTRY.metric("fill_rate")
        assert any("|" in c for c in lattice_combos(REGISTRY, metric, "1h"))


class TestViewChain:
    def test_hourly_reads_events_not_the_five_minute_table(self):
        """A coarser grain cannot recover two-way cells its source never stored.

        Chaining hourly off rollup_5m is cheaper and was the original design; it became wrong
        the moment the 5-minute tier went shallow. The bug it would cause is quiet -- rollup_1h
        would simply hold one-way cells only, and localization would find no candidates.
        """
        assert ROLLUP_SOURCE["1h"] == "events"
        views = {s.name: s.sql for s in view_ddl(DIMS)}
        assert "FROM ad_events" in views["mv_events_to_1h"]
        assert "mv_5m_to_1h" not in views

    def test_daily_chains_off_hourly_since_both_are_full_depth(self):
        assert ROLLUP_SOURCE["1d"] == "rollup_1h"
        views = {s.name: s.sql for s in view_ddl(DIMS)}
        assert "FROM rollup_1h" in views["mv_1h_to_1d"]

    def test_there_is_exactly_one_view_per_grain(self):
        assert len(view_ddl(DIMS)) == len(GRAINS)

    def test_five_minute_view_carries_no_pair_combos(self):
        sql = {s.name: s.sql for s in view_ddl(DIMS)}["mv_events_to_5m"]
        for a in DIMS:
            for b in DIMS:
                assert f"'{a}|{b}'" not in sql

    def test_hourly_view_carries_every_pair(self):
        sql = {s.name: s.sql for s in view_ddl(DIMS)}["mv_events_to_1h"]
        for a, b in [c for c in combos(DIMS, 2) if c[1] is not None]:
            assert f"'{a}|{b}'" in sql


class TestBackfillDoesNotDoubleCount:
    """The bug this class exists for cost a full nine-million-row load to find.

    A materialized view fires on inserts made by the backfill exactly as it does on live
    traffic. So `INSERT INTO rollup_1h` also writes rollup_1d through mv_1h_to_1d, and issuing
    a backfill statement for the daily grain as well wrote every row twice. Nothing local looks
    wrong afterwards -- both tables are populated, both are internally consistent, and only a
    cross-check against the fact table reveals the daily grain carrying exactly double.
    """

    def test_no_backfill_targets_a_table_a_view_already_cascades_into(self):
        targets = {s.name.removeprefix("backfill_") for s in backfill_statements(DIMS)}
        assert targets.isdisjoint(cascaded_grains())

    def test_every_grain_is_filled_exactly_once(self):
        written = [s.name.removeprefix("backfill_") for s in backfill_statements(DIMS)]
        written += cascaded_grains()
        assert sorted(written) == sorted(GRAINS)
        assert len(written) == len(set(written)), "a grain is written by two paths"

    def test_each_backfill_uses_the_same_select_as_its_view(self):
        """If backfilled history and streamed traffic disagree, the handover between them is a
        step change in the data -- the exact shape of the incidents this system hunts, so it
        would be detected, localized, and reported as a genuine finding."""
        views = {s.name: s.sql for s in view_ddl(DIMS)}
        for stmt in backfill_statements(DIMS):
            grain = stmt.name.removeprefix("backfill_")
            select = stmt.sql.split("\n", 1)[1].strip()
            assert select in views[view_name(grain)]


def _config(**overrides: str):
    env = {
        "CLICKHOUSE_HOST": "localhost",
        "CLICKHOUSE_PASSWORD": "",
        "LLM_ENABLED": "false",
        **overrides,
    }
    return load_config(ROOT / "config" / "verdict.yaml", env=env)


class TestAllStatements:
    def test_generates_without_error_and_names_are_unique(self):
        stmts = all_statements(_config(), REGISTRY)
        names = [s.name for s in stmts]
        assert len(names) == len(set(names))

    def test_ttl_is_absent_unless_enforced(self):
        """A historical corpus is older than a 7-day raw TTL, so enforcing by default would
        instruct ClickHouse to delete everything on the first background merge."""
        assert all("TTL" not in s.sql for s in all_statements(_config(), REGISTRY))

    def test_ttl_appears_when_enforcement_is_asked_for(self):
        sql = "\n".join(s.sql for s in all_statements(_config(RETENTION_ENFORCE="true"), REGISTRY))
        assert "TTL event_time + INTERVAL 7 DAY DELETE" in sql
