"""Turning a verdict into a durable case.

A case is written once and read many times, by consumers with very different needs: the
dashboard wants a headline, the drill-down wants every step, an operator six months from now
wants to know why a segment was cleared, and the recurrence check wants to know whether this
exact shape has been seen before. All of them read the same rows, so none of them can be told
a different story.

Two decisions in here are load-bearing.

The first is that a case records what was ruled out, not only what was accused. A verdict that
names a culprit and says nothing else is unfalsifiable; a verdict that also publishes what it
predicted the innocent segments would do, and what they actually did, can be checked by anyone
who doubts it. The exoneration ledger is written to ``case_candidates`` alongside the accused
for exactly that reason.

The second is that the fingerprint is deliberately coarser than the case id. The case id
identifies one detection in one run; the fingerprint identifies the *shape* of an incident, so
that the same failure recurring in November can be linked to its August predecessor even though
every number differs. Making them the same field would mean either never recognising a repeat
or treating two unrelated incidents as one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .db import ClickHouse
from .detect import CoverageGap, Finding
from .localize import Candidate, Localization
from .query import Counters, Segment, Window

log = logging.getLogger(__name__)


CASE_COLUMNS = (
    "case_id",
    "run_id",
    "detected_at",
    "metric",
    "grain",
    "window_start",
    "window_end",
    "direction",
    "observed",
    "expected",
    "relative_effect",
    "p_value",
    "dispersion",
    "verdict_kind",
    "segment",
    "segment_json",
    "confidence",
    "confidence_json",
    "gates_json",
    "impact_json",
    "narrative",
    "narrative_source",
    "narrative_model",
    "narrative_verified",
    "narrative_rejected",
    "narrative_prompt_tokens",
    "narrative_completion_tokens",
    "narrative_latency_ms",
    "fingerprint",
    "trace_id",
    "recurrence_of",
    "detector",
    "mode",
    "cells_tested",
)

CANDIDATE_COLUMNS = (
    "case_id",
    "candidate",
    "candidate_json",
    "depth",
    "observed",
    "expected",
    "predicted",
    "residual",
    "sufficiency",
    "minimality",
    "maximality",
    "holdout",
    "p_value",
    "status",
    "reason",
)

STEP_COLUMNS = (
    "case_id",
    "step_id",
    "parent_id",
    "ordinal",
    "name",
    "kind",
    "what",
    "why",
    "result",
    "sql",
    "duration_ms",
    "offset_ms",
    "span_id",
)

COVERAGE_COLUMNS = (
    "run_id",
    "metric",
    "grain",
    "window_start",
    "combo",
    "key_a",
    "key_b",
    "denominator",
    "required",
    "reason",
    "resolvable_effect",
)

RUN_COLUMNS = (
    "run_id",
    "started_at",
    "finished_at",
    "config_json",
    "git_sha",
    "trace_id",
    "cases_found",
    "status",
    "note",
)

# Metrics whose deviation converts to money without passing through another estimate. Everything
# else needs a chain, and a chained figure has to be labelled as one.
DIRECT_REVENUE_METRICS = frozenset({"ecpm", "rpr", "revenue"})


def _utc(value: datetime) -> datetime:
    """ClickHouse DateTime('UTC') columns reject an offset-aware value from another zone."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def direction_of(relative_effect: float) -> str:
    if relative_effect > 0:
        return "rise"
    if relative_effect < 0:
        return "fall"
    return "flat"


def fingerprint(metric: str, segment: Segment, direction: str) -> str:
    """A stable identity for the shape of an incident, independent of when it happened.

    Deliberately excludes magnitude, window and run. Two outages in the same segment of the same
    metric are the same problem recurring even when one costs twice as much as the other, and an
    operator who has seen the first is far better served by being told "this is the third time"
    than by being handed a fresh case with no history.
    """
    keys = "|".join(f"{k}={v}" for k, v in sorted(segment.keys))
    return hashlib.sha1(f"{metric}\x1f{keys}\x1f{direction}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Impact:
    """What the deviation cost, with its derivation attached.

    Every consumer of this number will quote it, so it travels with the assumptions that produced
    it. A revenue figure derived through two multiplications is not the same claim as one read
    off the ledger, and presenting them identically is how an estimate becomes a fact nobody can
    later challenge.
    """

    units: float
    unit: str
    revenue: float | None
    direct: bool
    basis: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": round(self.units, 4),
            "unit": self.unit,
            "revenue": None if self.revenue is None else round(self.revenue, 2),
            "direct": self.direct,
            "basis": list(self.basis),
        }


def estimate_impact(metric: str, observed: Counters, observed_value: float, expected_value: float) -> Impact:
    """Convert a rate deviation into units lost, and into money where the funnel permits it.

    The conversion is only as sound as the shortest path from the metric to revenue. Fill rate
    reaches money through two further rates, each measured on the affected segment during the
    incident itself, which is conservative in the right direction: if the incident also depressed
    those rates, the estimate understates the loss rather than inflating it.

    Click-through rate is left without a revenue figure on purpose. Revenue in this dataset
    accrues on impressions, not clicks, so any money attached to lost clicks would be a
    downstream advertiser-value argument this system has no data to make.
    """
    shortfall = expected_value - observed_value

    if metric == "fill_rate":
        lost_fills = shortfall * observed.requests
        render = observed.impressions / observed.fills if observed.fills else 0.0
        rev_per_impression = observed.revenue / observed.impressions if observed.impressions else 0.0
        return Impact(
            units=lost_fills,
            unit="fills",
            revenue=lost_fills * render * rev_per_impression,
            direct=False,
            basis=(
                f"{lost_fills:,.0f} fills short of expectation across {observed.requests:,} requests",
                f"carried to impressions at the segment's own render rate of {render:.4f}",
                f"valued at the segment's own revenue per impression of {rev_per_impression:.6f}",
            ),
        )

    if metric == "render_rate":
        lost_impressions = shortfall * observed.fills
        rev_per_impression = observed.revenue / observed.impressions if observed.impressions else 0.0
        return Impact(
            units=lost_impressions,
            unit="impressions",
            revenue=lost_impressions * rev_per_impression,
            direct=False,
            basis=(
                f"{lost_impressions:,.0f} impressions short across {observed.fills:,} fills",
                f"valued at the segment's own revenue per impression of {rev_per_impression:.6f}",
            ),
        )

    if metric == "ctr":
        lost_clicks = shortfall * observed.impressions
        return Impact(
            units=lost_clicks,
            unit="clicks",
            revenue=None,
            direct=False,
            basis=(
                f"{lost_clicks:,.0f} clicks short across {observed.impressions:,} impressions",
                "no revenue figure: revenue accrues on impressions in this dataset, not clicks",
            ),
        )

    if metric == "ecpm":
        lost = shortfall * observed.impressions / 1000.0
        return Impact(
            units=lost,
            unit="revenue",
            revenue=lost,
            direct=True,
            basis=(f"eCPM shortfall of {shortfall:.4f} over {observed.impressions:,} impressions",),
        )

    if metric == "rpr":
        lost = shortfall * observed.requests
        return Impact(
            units=lost,
            unit="revenue",
            revenue=lost,
            direct=True,
            basis=(f"revenue-per-request shortfall of {shortfall:.6f} over {observed.requests:,} requests",),
        )

    return Impact(units=shortfall, unit=metric, revenue=None, direct=False, basis=())


def _check_score(candidate: Candidate, name: str) -> float:
    """A check's score flattened for the columnar store.

    An unscored or absent check lands on 0.0 here, which the confidence model must never read as
    a failing score -- the authoritative three-state result lives in ``gates_json``. This column
    exists for sorting and filtering in SQL, not for arithmetic.
    """
    check = candidate.checks.get(name)
    if check is None or check.score is None:
        return 0.0
    return float(check.score)


def gates_of(localization: Localization) -> dict[str, Any]:
    """The three-state outcome of every check, preserved exactly as the localizer left it.

    Collapsing pass/fail/unknown into a number loses the distinction between a test that failed
    and a test that could not be run, and that distinction is the difference between "we looked
    and this is not it" and "we never looked".
    """
    accused = localization.accused
    if accused is None:
        return {"mode": localization.mode, "note": localization.note, "checks": {}}
    return {
        "mode": localization.mode,
        "note": localization.note,
        "checks": {
            name: {"state": check.state, "score": check.score, "detail": check.detail}
            for name, check in sorted(accused.checks.items())
        },
    }


@dataclass
class Case:
    """One detection, localized, scored and ready to be written."""

    case_id: str
    run_id: str
    detected_at: datetime
    finding: Finding
    localization: Localization
    impact: Impact
    fingerprint: str
    confidence_value: float = 0.0
    confidence_json: str = "{}"
    narrative: str = ""
    narrative_source: str = "template"
    narrative_model: str = ""
    narrative_verified: bool = False
    narrative_rejected: list[str] = field(default_factory=list)
    narrative_prompt_tokens: int = 0
    narrative_completion_tokens: int = 0
    narrative_latency_ms: int = 0
    trace_id: str = ""
    recurrence_of: str = ""
    # Size of the sweep this finding came out of. Held on the case rather than looked up from
    # the run, because a reader weighing one survivor of four thousand tests against one of
    # forty needs the denominator next to the claim, not a join away.
    cells_tested: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def detector(self) -> str:
        return self.finding.detector

    @property
    def mode(self) -> str:
        return self.localization.mode

    @property
    def segment(self) -> Segment:
        accused = self.localization.accused
        return accused.segment if accused is not None else self.finding.segment

    @property
    def verdict_kind(self) -> str:
        if self.localization.accused is not None:
            return "localized"
        if self.localization.mode == "no_data":
            return "no_data"
        if self.localization.candidates:
            return "unlocalized"
        return "undecomposed"

    def case_row(self) -> list[Any]:
        loc = self.localization
        test = self.finding.test
        accused = loc.accused
        observed = accused.observed_value if accused else test.observed
        expected = accused.expected_value if accused else test.expected
        effect = accused.relative_effect if accused else test.relative_effect
        return [
            self.case_id,
            self.run_id,
            _utc(self.detected_at),
            self.finding.metric,
            self.finding.window.grain,
            _utc(self.finding.window.start),
            _utc(self.finding.window.end),
            direction_of(effect),
            float(observed or 0.0),
            float(expected or 0.0),
            float(effect or 0.0),
            float(test.p_value),
            float(self.finding.phi),
            self.verdict_kind,
            self.segment.label(),
            _json(self.segment.as_dict()),
            float(self.confidence_value),
            self.confidence_json,
            _json(gates_of(loc)),
            _json(self.impact.to_dict()),
            self.narrative,
            self.narrative_source,
            self.narrative_model,
            1 if self.narrative_verified else 0,
            list(self.narrative_rejected),
            int(self.narrative_prompt_tokens),
            int(self.narrative_completion_tokens),
            int(self.narrative_latency_ms),
            self.fingerprint,
            self.trace_id,
            self.recurrence_of,
            self.detector,
            self.mode,
            int(self.cells_tested),
        ]

    def candidate_rows(self) -> list[list[Any]]:
        """Every candidate considered, accused and cleared alike.

        Writing only the accused would halve the storage and destroy the point: the cleared rows
        carry ``predicted`` and ``residual``, which is the record of what this system said an
        innocent segment would do before it knew whether it was right.
        """
        rows: list[list[Any]] = []
        for candidate in self.localization.candidates:
            rows.append(
                [
                    self.case_id,
                    candidate.segment.label(),
                    _json(candidate.segment.as_dict()),
                    candidate.segment.depth,
                    float(candidate.observed_value or 0.0),
                    float(candidate.expected_value or 0.0),
                    float(candidate.predicted_if_innocent or 0.0),
                    float(candidate.exoneration_residual or 0.0),
                    _check_score(candidate, "sufficiency"),
                    _check_score(candidate, "minimality"),
                    _check_score(candidate, "maximality"),
                    _check_score(candidate, "holdout"),
                    0.0,
                    candidate.status,
                    candidate.reason,
                ]
            )
        return rows

    def step_rows(self) -> list[list[Any]]:
        return [
            [
                step.get("case_id", self.case_id),
                step["step_id"],
                step["parent_id"],
                int(step["ordinal"]),
                step["name"],
                step["kind"],
                step["what"],
                step["why"],
                step["result"],
                step["sql"],
                int(step["duration_ms"]),
                int(step.get("offset_ms", 0)),
                step["span_id"],
            ]
            for step in self.steps
        ]


def build_case(
    finding: Finding,
    localization: Localization,
    *,
    run_id: str,
    confidence: Any | None = None,
    narration: Any | None = None,
    trace_id: str = "",
    steps: list[dict[str, Any]] | None = None,
    detected_at: datetime | None = None,
    cells_tested: int = 0,
) -> Case:
    """Assemble a case from the parts each stage produced.

    ``confidence`` and ``narration`` are read defensively rather than typed, because both are
    optional stages: a case must still be storable when the model is switched off or the scorer
    is unavailable, and a persistence layer that raises in that situation would throw away a
    perfectly good investigation over a missing sentence.
    """
    accused = localization.accused
    observed = accused.observed if accused is not None else finding.observed_counters
    observed_value = (accused.observed_value if accused else finding.test.observed) or 0.0
    expected_value = (accused.expected_value if accused else finding.test.expected) or 0.0
    effect = accused.relative_effect if accused else finding.test.relative_effect

    segment = accused.segment if accused is not None else finding.segment
    case_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{run_id}/{finding.metric}/{finding.window.start:%Y%m%d%H%M}/{segment.label()}",
    ).hex

    return Case(
        case_id=case_id,
        run_id=run_id,
        detected_at=detected_at or datetime.now(UTC),
        finding=finding,
        localization=localization,
        impact=estimate_impact(finding.metric, observed, observed_value, expected_value),
        fingerprint=fingerprint(finding.metric, segment, direction_of(effect)),
        confidence_value=float(getattr(confidence, "value", 0.0) or 0.0),
        confidence_json=_json(_confidence_payload(confidence)),
        narrative=str(getattr(narration, "text", "") or ""),
        narrative_source=str(getattr(narration, "source", "template") or "template"),
        narrative_model=str(getattr(narration, "model", "") or ""),
        narrative_verified=bool(getattr(narration, "verified", False)),
        narrative_rejected=[str(f) for f in (getattr(narration, "unsupported", None) or [])],
        narrative_prompt_tokens=int(getattr(narration, "prompt_tokens", 0) or 0),
        narrative_completion_tokens=int(getattr(narration, "completion_tokens", 0) or 0),
        narrative_latency_ms=int(getattr(narration, "latency_ms", 0) or 0),
        cells_tested=int(cells_tested),
        trace_id=trace_id,
        steps=steps or [],
    )


def _confidence_payload(confidence: Any | None) -> dict[str, Any]:
    if confidence is None:
        return {}
    components = []
    for component in getattr(confidence, "components", []) or []:
        components.append(
            {
                "name": getattr(component, "name", ""),
                "score": getattr(component, "score", None),
                "weight": getattr(component, "weight", None),
                "state": getattr(component, "state", "scored"),
                "detail": getattr(component, "detail", ""),
            }
        )
    return {
        "value": getattr(confidence, "value", None),
        "components": components,
        "components_scored": getattr(confidence, "components_scored", None),
        "components_total": getattr(confidence, "components_total", None),
        "publishable": getattr(confidence, "publishable", None),
        "caveat": getattr(confidence, "caveat", ""),
    }


class CaseStore:
    """Reads and writes everything the analyst concluded."""

    def __init__(self, ch: ClickHouse) -> None:
        self.ch = ch

    def open_run(self, run_id: str, *, config_json: str = "{}", git_sha: str = "", trace_id: str = "") -> None:
        now = _utc(datetime.now(UTC))
        self.ch.insert(
            "runs",
            [[run_id, now, now, config_json, git_sha, trace_id, 0, "running", ""]],
            RUN_COLUMNS,
            name="insert_run_open",
        )

    def close_run(
        self,
        run_id: str,
        *,
        cases_found: int,
        status: str = "complete",
        note: str = "",
        config_json: str = "{}",
        git_sha: str = "",
        trace_id: str = "",
        started_at: datetime | None = None,
    ) -> None:
        """Close the run out.

        ``runs`` is a ReplacingMergeTree keyed on run_id, so this supersedes the open row rather
        than adding a second one. Recording the terminal status matters even when it is a
        failure: a run that died halfway leaves cases in the table, and without this row there is
        no way to tell those apart from a run that genuinely found nothing.
        """
        now = _utc(datetime.now(UTC))
        self.ch.insert(
            "runs",
            [
                [
                    run_id,
                    _utc(started_at) if started_at else now,
                    now,
                    config_json,
                    git_sha,
                    trace_id,
                    int(cases_found),
                    status,
                    note,
                ]
            ],
            RUN_COLUMNS,
            name="insert_run_close",
        )

    def find_recurrence(self, fingerprint_value: str, *, before: datetime, exclude_run: str = "") -> str:
        """The most recent earlier case with this shape, or empty when this is the first.

        Restricted to strictly earlier detections so that two cases produced by the same run
        cannot each claim to be a recurrence of the other.
        """
        sql = """
            SELECT case_id
            FROM cases
            WHERE fingerprint = {fp:String}
              AND detected_at < {before:DateTime}
              AND run_id != {run:String}
            ORDER BY detected_at DESC
            LIMIT 1
        """
        rows = self.ch.query(
            sql,
            {"fp": fingerprint_value, "before": _utc(before), "run": exclude_run},
            name="find_recurrence",
        )
        return str(rows[0][0]) if rows else ""

    def write_case(self, case: Case, *, link_recurrence: bool = True) -> None:
        if link_recurrence and not case.recurrence_of:
            try:
                case.recurrence_of = self.find_recurrence(
                    case.fingerprint, before=case.detected_at, exclude_run=case.run_id
                )
            except Exception as exc:  # noqa: BLE001 - a missing back-link must not lose the case
                log.warning("Recurrence lookup failed for %s: %s", case.case_id, exc)

        self.ch.insert("cases", [case.case_row()], CASE_COLUMNS, name="insert_case")

        candidates = case.candidate_rows()
        if candidates:
            self.ch.insert("case_candidates", candidates, CANDIDATE_COLUMNS, name="insert_candidates")

        steps = case.step_rows()
        if steps:
            self.ch.insert("case_steps", steps, STEP_COLUMNS, name="insert_steps")

    def write_coverage(self, run_id: str, gaps: list[CoverageGap], window: Window) -> int:
        """Record what could not be tested.

        This is the least glamorous table and the most important one for honesty. Without it a
        run that tested four percent of the lattice and a run that tested all of it produce
        indistinguishable output, and silence gets read as an all-clear.

        ``resolvable_effect`` is stored as a negative sentinel when the cell could not have
        resolved even a near-total collapse. Storing NULL would push the distinction into a
        nullable column that every downstream aggregate then has to special-case, and the
        sentinel is unambiguous because a resolvable effect is a magnitude and can never
        legitimately be below zero.
        """
        rows = []
        for gap in gaps:
            key_a, key_b = gap.segment.combo_keys
            rows.append(
                [
                    run_id,
                    gap.metric,
                    window.grain,
                    _utc(window.start),
                    gap.segment.combo,
                    key_a,
                    key_b,
                    int(max(0, gap.denominator)),
                    int(max(0, gap.required)),
                    gap.reason,
                    -1.0 if gap.resolvable_effect is None else float(gap.resolvable_effect),
                ]
            )
        if rows:
            self.ch.insert("coverage_ledger", rows, COVERAGE_COLUMNS, name="insert_coverage")
        return len(rows)
