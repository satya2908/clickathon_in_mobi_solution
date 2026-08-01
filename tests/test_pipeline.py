"""What a case carries with it, and what it leaves behind."""

from __future__ import annotations
    
from verdict.pipeline import for_metric
from verdict.trace import Step


class TestACaseCarriesItsOwnMetricsSweep:
    """A ten-metric run emits a detection span per metric per lattice combination. Copying all
    of them onto every case made four thousand rows for nine cases, most of it a fill_rate case
    carrying the CTR scan -- spans that are not evidence for that verdict."""

    def _steps(self) -> list[Step]:
        return [
            Step(step_id="1", parent_id="", ordinal=1, name="investigate", kind="pipeline"),
            Step(step_id="2", parent_id="1", ordinal=2, name="detect", kind="detector"),
            Step(step_id="3", parent_id="2", ordinal=3, name="temporal:fill_rate:__all__", kind="detector"),
            Step(step_id="4", parent_id="2", ordinal=4, name="temporal:ctr:region", kind="detector"),
            Step(step_id="5", parent_id="2", ordinal=5, name="structural:fill_rate:country|region", kind="detector"),
            Step(step_id="6", parent_id="1", ordinal=6, name="correct", kind="statistics"),
        ]

    def test_another_metrics_scan_is_left_out(self):
        kept = {s.name for s in for_metric(self._steps(), "fill_rate")}
        assert "temporal:ctr:region" not in kept

    def test_its_own_scan_is_kept(self):
        kept = {s.name for s in for_metric(self._steps(), "fill_rate")}
        assert "temporal:fill_rate:__all__" in kept
        assert "structural:fill_rate:country|region" in kept

    def test_the_stages_that_hold_the_tree_together_are_always_kept(self):
        """Dropping a stage with no metric in its name would orphan everything beneath it."""
        kept = {s.name for s in for_metric(self._steps(), "ctr")}
        assert {"investigate", "detect", "correct"} <= kept

    def test_every_kept_step_still_has_its_parent(self):
        for metric in ("fill_rate", "ctr", "revenue"):
            kept = for_metric(self._steps(), metric)
            ids = {s.step_id for s in kept}
            assert all(not s.parent_id or s.parent_id in ids for s in kept), metric

    def test_a_metric_that_never_ran_keeps_only_the_scaffolding(self):
        assert [s.name for s in for_metric(self._steps(), "revenue")] == [
            "investigate",
            "detect",
            "correct",
        ]
