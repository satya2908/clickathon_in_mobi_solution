"""Synthetic incident injection: the instrument every other claim in this system is measured
against.

The rest of the codebase asserts that it finds root causes. Nothing in it can demonstrate that,
because the only incidents available are the ones it was built while looking at, and recovering
those proves little more than that the author remembered them. Planting incidents of known
shape, size, segment and window turns recall, precision and localization accuracy into
quantities that can be computed instead of claimed.

Two design commitments follow from that, and they are the entire reason this module is not
simply a helper that scales a few numbers.

**It plants shapes the detectors were not designed for.** A harness that only injects
one-way marginal drops measures how well the system handles the case it was written for and
reports a flattering number that generalises to nothing. The catalogue therefore includes
high-cardinality keys the rollup lattice does not carry, gradual drift that a trailing baseline
absorbs by construction, and spikes shorter than the finest stored grain. Those are expected to
be missed, and each plan says so up front, so a miss is scored as a known blind spot rather
than discovered as a surprise in front of a judge.

**It plants nothing at all.** ``clean`` is a control: the data comes back row for row identical
and the answer key says the correct number of findings is zero. Without controls the false
positive rate is unmeasurable, and a system that reports three confident incidents on untouched
data looks exactly like a system that works.

Two smaller decisions are worth stating because they look like omissions.

The metric formulas are restated here rather than imported from the metric registry. If the
answer key and the detector both divide by the same wrong denominator, the injected incident
and the resulting finding agree and the evaluation scores a broken system as correct. A test
cross-checks these four formulas against ``config/metrics.yaml``, so drift surfaces as a
disagreement between two independent statements rather than being cancelled out by a shared
one.

Nothing here touches ClickHouse. Injection operates on an event set in memory -- a pyarrow
Table or a list of row dicts -- so an evaluation run is reproducible from a seed, and a harness
bug cannot corrupt a loaded corpus that took twenty minutes to ingest.
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

log = logging.getLogger(__name__)

KIND_MAIN_EFFECT = "main_effect"
KIND_INTERACTION = "interaction"
KIND_COMPENSATING_PAIR = "compensating_pair"
KIND_HIGH_CARDINALITY = "high_cardinality"
KIND_SLOW_DRIFT = "slow_drift"
KIND_SPIKE = "spike"
KIND_CLEAN = "clean"

KINDS = (
    KIND_MAIN_EFFECT,
    KIND_INTERACTION,
    KIND_COMPENSATING_PAIR,
    KIND_HIGH_CARDINALITY,
    KIND_SLOW_DRIFT,
    KIND_SPIKE,
    KIND_CLEAN,
)

# numerator field, denominator field, scale. Restated rather than imported; see the module
# docstring for why an independent statement is the point.
_FORMULA = {
    "fill_rate": ("fills", "requests", 1.0),
    "render_rate": ("impressions", "fills", 1.0),
    "ctr": ("clicks", "impressions", 1.0),
    "ecpm": ("revenue", "impressions", 1000.0),
}

MOVABLE_METRICS = tuple(_FORMULA)

# Columns the injector writes. Everything else is passed through untouched, which is what lets
# the Arrow path hand back the original buffers for any column it did not need to rewrite.
_FUNNEL_COLUMNS = ("is_filled", "is_impression", "is_click", "revenue")

# Dimensions that only exist once a request was filled, because advertiser_id is empty on
# unfilled rows. Injecting a fill-rate incident on one of these is not a hard error anywhere
# downstream, which is exactly the problem: clearing is_filled removes the row from the segment
# altogether, so the segment's fill rate stays pinned at 1.0 and the answer key records a drop
# that no query can ever see.
_POST_FILL_DIMENSIONS = frozenset({"advertiser_id", "vertical", "campaign_type"})

# Metrics whose denominator is every request, and therefore cannot be sliced by the above.
_REQUEST_GRAINED_METRICS = frozenset({"fill_rate"})

# Keys the rollup lattice deliberately excludes. `high_cardinality` must name one of these or
# it is not testing the blind spot it claims to test.
_HIGH_CARDINALITY_COLUMNS = frozenset({"app_id", "advertiser_id", "geo_device_id"})

# A spike exists to probe whether the stored grain can resolve a short move at all. The finest
# rollup here is five minutes, so anything approaching an hour is a main effect wearing a
# different label and would report a pass that says nothing about grain sensitivity.
_SPIKE_MAX_SECONDS = 3600

# Why each shape is expected to be missed. An empty entry means the system should catch it.
# Publishing the expectation with the plan is what separates a known limitation from a
# regression: an unexpected miss is a bug, an expected one is a documented boundary.
_BLIND_SPOTS = {
    KIND_HIGH_CARDINALITY: (
        "app_id, advertiser_id and geo_device_id are outside the rollup lattice, so no cell "
        "exists at this grain for the detector to test. A miss here is the stated limit of "
        "the storage shape, not a detection failure."
    ),
    KIND_SLOW_DRIFT: (
        "The baseline is a trailing average over recent weeks, so a decline spread across "
        "several days is partly absorbed into the expectation it would have to be measured "
        "against. Expect this to be understated or missed entirely."
    ),
    KIND_SPIKE: (
        "The finest stored grain is five minutes and carries one-way cells only. A move "
        "shorter than one bucket is diluted by the untouched remainder of that bucket before "
        "any detector sees it."
    ),
    KIND_CLEAN: (
        "Nothing was injected. The correct output is no findings at all; anything reported "
        "here is a false positive."
    ),
}

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class InjectionError(ValueError):
    """Raised for a specification that cannot mean what it says.

    Every one of these is preferred to a silent partial injection, because the answer key is
    written from the plan: a spec that quietly does something other than what it describes
    produces an evaluation that is wrong in the direction of looking correct.
    """


@dataclass(frozen=True)
class Totals:
    """Additive counters over a set of rows, the only form a metric is computed from."""

    requests: int = 0
    fills: int = 0
    impressions: int = 0
    clicks: int = 0
    revenue: float = 0.0


def metric_value(totals: Totals, metric: str) -> float | None:
    """The metric's value, or None when its denominator is empty.

    None rather than zero, deliberately. A segment with no impressions does not have a CTR of
    zero, and feeding a fabricated zero into a realised-magnitude calculation would record a
    100% drop that never happened.
    """
    if metric not in _FORMULA:
        raise InjectionError(f"Unknown metric {metric!r}. Known: {', '.join(MOVABLE_METRICS)}")
    numerator, denominator, scale = _FORMULA[metric]
    den = getattr(totals, denominator)
    if not den:
        return None
    return getattr(totals, numerator) / den * scale


@dataclass(frozen=True)
class DimensionIndex:
    """Resolves dimensions the fact table does not carry.

    ``ad_events`` stores geo_device_id and app_id, not country or publisher_tier; in ClickHouse
    those are dictGet lookups. An injector that could only match on physical columns would be
    unable to plant an incident on any of the seven dimensions the lattice is actually built
    from, which is most of the catalogue. This holds the same mapping in memory so the segment
    a spec names is the segment the rollup will later group by.
    """

    lookups: dict[str, tuple[str, dict[str, str]]] = field(default_factory=dict)

    @property
    def dimensions(self) -> list[str]:
        return sorted(self.lookups)

    @classmethod
    def from_csv(cls, path: str | Path, key_column: str) -> DimensionIndex:
        """Build an index from one dimension CSV, keyed by ``key_column``.

        Reads the file as-is with no type coercion: these values are compared against fact
        table strings, and a column silently parsed as an integer would match nothing while
        reporting a perfectly healthy zero-row injection.
        """
        p = Path(path)
        with p.open(newline="") as fh:
            reader = csv.DictReader(fh)
            names = list(reader.fieldnames or [])
            if key_column not in names:
                raise InjectionError(
                    f"{p.name} has no column {key_column!r}; found {names}. Refusing to build "
                    "an index that would resolve every lookup to a miss."
                )
            attributes = [n for n in names if n != key_column]
            tables: dict[str, dict[str, str]] = {a: {} for a in attributes}
            rows = 0
            for row in reader:
                key = row[key_column]
                for a in attributes:
                    tables[a][key] = row[a]
                rows += 1

        if not rows:
            raise InjectionError(f"{p.name} parsed to zero rows")
        return cls({a: (key_column, tables[a]) for a in attributes})

    def merged(self, other: DimensionIndex) -> DimensionIndex:
        clash = set(self.lookups) & set(other.lookups)
        if clash:
            raise InjectionError(
                f"Dimension(s) {sorted(clash)} are defined by two different key columns. One "
                "of them would win arbitrarily and half the injections would target the wrong "
                "rows."
            )
        return DimensionIndex({**self.lookups, **other.lookups})


@dataclass(frozen=True)
class IncidentSpec:
    """What to plant. The request, not the outcome."""

    kind: str
    metric: str
    where: dict[str, str]
    start: datetime
    end: datetime
    magnitude: float
    seed: int = 0
    note: str = ""
    # Only ``compensating_pair`` uses this: the segment that moves the other way. It cannot be
    # inferred, because which segment is a plausible counterweight is a fact about the data and
    # the injector never reads the data before planning.
    counterpart: dict[str, str] | None = None
    # Only ``slow_drift`` uses this. Zero picks one step per day of the window, which is the
    # granularity a daily baseline would have to chase.
    steps: int = 0


@dataclass(frozen=True)
class Edit:
    """One segment-and-window move. Every shape compiles to a list of these, which is what
    keeps a seven-entry catalogue from becoming seven code paths that drift apart."""

    where: tuple[tuple[str, str], ...]
    start_ms: int
    end_ms: int
    magnitude: float
    label: str
    # Index of an earlier edit whose numerator movement this one must cancel exactly. Expressed
    # as a reference rather than a magnitude because the cancellation has to be in counted
    # units: two segments that each move by 30% of their own baseline leave a residue in the
    # grand total unless they happen to be the same size.
    mirror_of: int | None = None


@dataclass(frozen=True)
class IncidentPlan:
    """The answer key. What a correct system should say about this data, written before the
    data is touched so it cannot be adjusted to match whatever came out."""

    spec: IncidentSpec
    expected_segment: dict[str, str]
    expected_metric: str
    expected_direction: str
    rows_affected: int
    description: str
    counterpart_segment: dict[str, str] | None = None
    expected_detectable: bool = True
    blind_spot: str = ""
    edits: tuple[Edit, ...] = ()
    # Which edit defines the headline magnitude. The last slice for a drift, since a drift is
    # specified by where it ends up; the first for everything else.
    reference_edit: int = 0
    realised_magnitude: float | None = None
    applied: bool = False

    def with_outcome(self, report: InjectionReport) -> IncidentPlan:
        """Fold what actually happened back into the key.

        ``rows_affected`` and the realised magnitude cannot be known at plan time -- both are
        properties of the data, and planning happens before the data is read. Scoring against
        the requested magnitude when the delivered one differed measures accuracy against a
        number that was never true of the corpus.
        """
        return IncidentPlan(
            spec=self.spec,
            expected_segment=self.expected_segment,
            expected_metric=self.expected_metric,
            expected_direction=self.expected_direction,
            rows_affected=report.rows_changed,
            description=self.description,
            counterpart_segment=self.counterpart_segment,
            expected_detectable=self.expected_detectable,
            blind_spot=self.blind_spot,
            edits=self.edits,
            reference_edit=self.reference_edit,
            realised_magnitude=report.realised_magnitude,
            applied=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable through ``json.dumps`` with no custom encoder.

        ``applied`` is in here on purpose. A key written from a plan that never went through
        ``apply`` carries a rows_affected of zero and a null realised magnitude, and scoring
        against it would silently mark every injected incident as a system miss. The harness
        can assert on one flag instead of noticing that the numbers look odd.
        """
        return {
            "kind": self.spec.kind,
            "metric": self.spec.metric,
            "where": dict(self.spec.where),
            "counterpart": dict(self.counterpart_segment) if self.counterpart_segment else None,
            "start": _isoformat(self.spec.start),
            "end": _isoformat(self.spec.end),
            "requested_magnitude": self.spec.magnitude,
            "realised_magnitude": self.realised_magnitude,
            "seed": self.spec.seed,
            "note": self.spec.note,
            "expected_segment": dict(self.expected_segment),
            "expected_metric": self.expected_metric,
            "expected_direction": self.expected_direction,
            "expected_detectable": self.expected_detectable,
            "blind_spot": self.blind_spot,
            "rows_affected": self.rows_affected,
            "applied": self.applied,
            "description": self.description,
            "windows": [
                {
                    "where": dict(e.where),
                    "start": _isoformat(_from_millis(e.start_ms)),
                    "end": _isoformat(_from_millis(e.end_ms)),
                    "magnitude": e.magnitude,
                    "mirror_of": e.mirror_of,
                }
                for e in self.edits
            ],
        }


@dataclass(frozen=True)
class InjectionReport:
    """What the injection actually did, as distinct from what it was asked to do."""

    rows_examined: int
    rows_matched: int
    rows_changed: int
    realised_magnitude: float
    funnel_violations: int
    # How many matched rows carried a flag that could be flipped in the requested direction.
    # This is the number that explains a shortfall: a segment with eleven filled requests
    # cannot deliver a 35% drop in anything, and without this the report says only that it
    # under-delivered.
    rows_eligible: int = 0
    # Relative movement of segments that must NOT move, keyed by label. Populated for the
    # shapes that make a claim about invisibility: an interaction claims both one-way marginals
    # stay flat, a compensating pair claims the grand total does. Measuring the claim rather
    # than asserting it is what makes those two shapes usable as evidence.
    marginal_shift: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def plan(spec: IncidentSpec) -> IncidentPlan:
    """Validate a spec and compile it into edits and an answer key.

    No data is read here. Everything this checks is a contradiction internal to the spec, and
    catching those before an evaluation run is the difference between a wasted afternoon and an
    error message.
    """
    if spec.kind not in KINDS:
        raise InjectionError(f"Unknown kind {spec.kind!r}. Known: {', '.join(KINDS)}")
    if spec.metric not in _FORMULA:
        raise InjectionError(
            f"Cannot move {spec.metric!r}. This module edits individual events, so it can only "
            f"move metrics that are a ratio of funnel counters: {', '.join(MOVABLE_METRICS)}."
        )
    if spec.start >= spec.end:
        raise InjectionError(
            f"Window start {spec.start} is not before end {spec.end}; no row can match it."
        )

    if spec.kind == KIND_CLEAN:
        return _plan_clean(spec)

    if not spec.where:
        raise InjectionError(
            f"Kind {spec.kind!r} needs a segment. An empty `where` moves the entire population, "
            "which no localizer can name and which the clean control already covers as the "
            "no-op case."
        )
    if spec.magnitude == 0.0:
        raise InjectionError("A magnitude of zero changes nothing. Use kind='clean' for that.")
    if spec.magnitude <= -1.0:
        raise InjectionError(
            f"Magnitude {spec.magnitude} would take the metric to zero or below. Rates are "
            "bounded at zero, so anything at or past -1.0 cannot be delivered or measured."
        )

    for segment in (spec.where, spec.counterpart or {}):
        _reject_illegal_slice(spec.metric, segment)

    start_ms, end_ms = _to_millis(spec.start), _to_millis(spec.end)
    where = _freeze(spec.where)
    edits: tuple[Edit, ...]
    counterpart: dict[str, str] | None = None
    reference = 0

    if spec.kind == KIND_INTERACTION:
        if len(spec.where) < 2:
            raise InjectionError(
                "An interaction is a cell at the intersection of two dimensions. With one "
                "dimension it is a main effect, and it would show up in that dimension's "
                "marginal, which is the opposite of what this shape exists to test."
            )
        edits = (Edit(where, start_ms, end_ms, spec.magnitude, "cell"),)

    elif spec.kind == KIND_HIGH_CARDINALITY:
        named = set(spec.where) & _HIGH_CARDINALITY_COLUMNS
        if not named:
            raise InjectionError(
                f"high_cardinality must target one of {sorted(_HIGH_CARDINALITY_COLUMNS)}; got "
                f"{sorted(spec.where)}. The shape exists to plant an incident the lattice "
                "cannot express, and a lattice dimension would simply be a main effect."
            )
        edits = (Edit(where, start_ms, end_ms, spec.magnitude, "entity"),)

    elif spec.kind == KIND_COMPENSATING_PAIR:
        if not spec.counterpart:
            raise InjectionError(
                "compensating_pair needs `counterpart`: the segment that moves the other way. "
                "Without it this is a main effect and the grand total moves, which is the one "
                "thing this shape must not do."
            )
        if _freeze(spec.counterpart) == where:
            raise InjectionError(
                "The counterpart is the same segment as the target, so the two moves would "
                "cancel inside one population and nothing would change anywhere."
            )
        counterpart = dict(spec.counterpart)
        edits = (
            Edit(where, start_ms, end_ms, spec.magnitude, "primary"),
            Edit(
                _freeze(spec.counterpart), start_ms, end_ms, -spec.magnitude, "counterpart",
                mirror_of=0,
            ),
        )

    elif spec.kind == KIND_SLOW_DRIFT:
        edits = _drift_edits(spec, where, start_ms, end_ms)
        reference = len(edits) - 1

    elif spec.kind == KIND_SPIKE:
        seconds = (end_ms - start_ms) / 1000.0
        if seconds >= _SPIKE_MAX_SECONDS:
            raise InjectionError(
                f"A spike lasting {seconds / 60:.0f} minutes is a step change at every grain "
                "this system stores, so it would test nothing about grain sensitivity. Keep it "
                f"under {_SPIKE_MAX_SECONDS // 60} minutes or use kind='main_effect'."
            )
        edits = (Edit(where, start_ms, end_ms, spec.magnitude, "spike"),)

    else:
        edits = (Edit(where, start_ms, end_ms, spec.magnitude, "segment"),)

    blind_spot = _BLIND_SPOTS.get(spec.kind, "")
    return IncidentPlan(
        spec=spec,
        expected_segment=dict(spec.where),
        expected_metric=spec.metric,
        expected_direction="fall" if spec.magnitude < 0 else "rise",
        rows_affected=0,
        description=_describe(spec, edits, blind_spot),
        counterpart_segment=counterpart,
        expected_detectable=not blind_spot,
        blind_spot=blind_spot,
        edits=edits,
        reference_edit=reference,
    )


def apply(
    table: Any, incident: IncidentPlan, *, dimensions: DimensionIndex | None = None
) -> tuple[Any, InjectionReport]:
    """Inject into a pyarrow Table, returning a new Table and what was done to it.

    Only the funnel columns are rebuilt. Every other column is handed back as the original
    Arrow array, which is both cheap and a guarantee: an injection cannot perturb a timestamp
    or a dimension key even by a rounding error, so the segment a row belonged to before is the
    segment it belongs to after. Rebuilt columns keep their original Arrow type, because a
    uint8 flag widened to int64 on the way through would be rejected or silently coerced at
    insert time, long after this function reported success.
    """
    for column in _FUNNEL_COLUMNS:
        if column not in table.schema.names:
            raise InjectionError(
                f"Table has no column {column!r}. The funnel invariant cannot be enforced on a "
                "table that does not carry the whole funnel, and treating the column as absent "
                "zeros would make the check vacuous rather than failing."
            )

    def read(name: str) -> list[Any] | None:
        if name not in table.schema.names:
            return None
        return table.column(name).to_pylist()

    events = _Events(table.num_rows, read, lambda: _arrow_millis(table.column("event_time")))
    report = _inject(events, incident, dimensions)

    out = table
    for name in sorted(events.changed):
        index = out.schema.get_field_index(name)
        original = out.schema.field(index)
        out = out.set_column(index, original, pa.array(events.column(name), type=original.type))
    return out, report


def apply_rows(
    rows: Sequence[Mapping[str, Any]],
    incident: IncidentPlan,
    *,
    dimensions: DimensionIndex | None = None,
) -> tuple[list[dict[str, Any]], InjectionReport]:
    """The same injection over a list of row dicts.

    This is the primary path in the sense that matters: ``apply`` converts to columns and calls
    the identical core, so an Arrow injection and a dict injection cannot disagree. Two
    implementations of the funnel cascade would be two chances to get it wrong, and the
    difference would only be visible in whichever one the tests did not exercise.

    Rows are copied rather than mutated. A harness that injects three variants from one loaded
    corpus would otherwise be comparing each run against an input the previous run had already
    edited.
    """
    data = [dict(r) for r in rows]
    available = set(data[0]) if data else set(_FUNNEL_COLUMNS)
    missing = [c for c in _FUNNEL_COLUMNS if c not in available]
    if data and missing:
        raise InjectionError(f"Rows are missing funnel column(s) {missing}")

    def read(name: str) -> list[Any] | None:
        if name not in available:
            return None
        return [r.get(name) for r in data]

    events = _Events(
        len(data), read, lambda: [_to_millis(r["event_time"]) for r in data]
    )
    report = _inject(events, incident, dimensions)

    for name in events.changed:
        column = events.column(name)
        for i, row in enumerate(data):
            row[name] = column[i]
    return data, report


class _Events:
    """Column-wise view over an event set, shared by both entry points.

    Columns are materialised on demand and only copied when something is about to write to
    them. That is what lets the Arrow path leave untouched columns as their original buffers,
    and it keeps a CTR injection from paying to convert nine million revenue values it will
    never look at.
    """

    def __init__(self, n: int, read: Any, times: Any) -> None:
        self.n = n
        self._read = read
        self._times_source = times
        self._times: list[int] | None = None
        self._columns: dict[str, list[Any] | None] = {}
        self.changed: set[str] = set()

    @property
    def times(self) -> list[int]:
        if self._times is None:
            self._times = self._times_source()
        return self._times

    def column(self, name: str) -> list[Any] | None:
        if name not in self._columns:
            self._columns[name] = self._read(name)
        return self._columns[name]

    def require(self, name: str) -> list[Any]:
        values = self.column(name)
        if values is None:
            raise InjectionError(f"Event set has no column {name!r}")
        return values

    def editable(self, name: str) -> list[Any]:
        values = self.require(name)
        if name not in self.changed:
            values = list(values)
            self._columns[name] = values
            self.changed.add(name)
        return values


@dataclass
class _Outcome:
    eligible: int = 0
    changed: int = 0
    realised: float = 0.0
    delta: float = 0.0
    notes: list[str] = field(default_factory=list)


def _inject(
    events: _Events, incident: IncidentPlan, dimensions: DimensionIndex | None
) -> InjectionReport:
    """Run every edit in the plan against one event set.

    Edits run in order because a mirrored edit needs the realised movement of the one it
    mirrors, and that is only known after it has run.
    """
    violations_before = _count_violations(events)
    notes: list[str] = []
    if violations_before:
        notes.append(
            f"{violations_before:,} rows already violated the funnel before injection. They are "
            "counted in funnel_violations and were not introduced here."
        )

    # Segment membership never changes -- no edit writes to a dimension key or to event_time --
    # so a segment is resolved once and reused by every edit that names it. A drift compiled
    # into twenty-four daily slices would otherwise rescan the whole corpus twenty-four times.
    cache: dict[tuple[tuple[str, str], ...], list[int]] = {}

    def segment_rows(where: tuple[tuple[str, str], ...]) -> list[int]:
        if where not in cache:
            cache[where] = _match_segment(events, where, dimensions)
        return cache[where]

    probes = [(label, segment_rows(where), a, b) for label, where, a, b in _probes(incident)]
    probe_before = {
        label: _totals(events, _within(events, rows, a, b)) for label, rows, a, b in probes
    }

    matched = bytearray(events.n)
    outcomes: list[_Outcome] = []
    for edit in incident.edits:
        rows = _within(events, segment_rows(edit.where), edit.start_ms, edit.end_ms)
        for i in rows:
            matched[i] = 1
        units = None
        if edit.mirror_of is not None:
            # The movement that was delivered, not the one that was asked for. A primary that
            # ran out of eligible rows still cancels exactly, where mirroring the request would
            # leave the difference sitting in the grand total as a residual incident nobody
            # planted and the answer key does not mention.
            units = -outcomes[edit.mirror_of].delta
        outcomes.append(
            _run_edit(events, incident.spec.metric, edit, rows, incident.spec.seed, units)
        )

    marginal_shift: dict[str, float] = {}
    for label, rows, a, b in probes:
        before = probe_before[label]
        after = _totals(events, _within(events, rows, a, b))
        shift = _relative_change(
            metric_value(before, incident.spec.metric), metric_value(after, incident.spec.metric)
        )
        marginal_shift[label] = shift

    for outcome in outcomes:
        notes.extend(outcome.notes)

    violations_after = _count_violations(events)
    if violations_after > violations_before:
        raise InjectionError(
            f"Injection introduced {violations_after - violations_before:,} funnel violations. "
            "The output describes events that could not have happened, so every metric "
            "computed from it would be meaningless. This is a bug in the injector."
        )

    realised = outcomes[incident.reference_edit].realised if outcomes else 0.0
    return InjectionReport(
        rows_examined=events.n,
        rows_matched=sum(matched),
        rows_changed=sum(o.changed for o in outcomes),
        realised_magnitude=realised,
        funnel_violations=violations_after,
        rows_eligible=sum(o.eligible for o in outcomes),
        marginal_shift=marginal_shift,
        notes=tuple(notes),
    )


def _run_edit(
    events: _Events,
    metric: str,
    edit: Edit,
    rows: list[int],
    seed: int,
    units: float | None,
) -> _Outcome:
    outcome = _Outcome()
    before = _totals(events, rows)
    value_before = metric_value(before, metric)
    if value_before is None:
        outcome.notes.append(
            f"{edit.label}: the segment has no {_FORMULA[metric][1]} in this window, so "
            f"{metric} is undefined and there is nothing to move."
        )
        return outcome

    # Seeded per edit rather than once per plan. Two edits sharing one stream would draw
    # correlated row sets whenever their eligible populations line up, and a compensating pair
    # that always picks structurally similar rows on both sides is not the independent
    # counterweight the shape claims to be.
    rng = random.Random(f"{seed}:{edit.label}")

    if metric == "ecpm":
        _move_ecpm(events, rows, before, edit, units, outcome)
    else:
        _move_rate(events, metric, rows, before, edit, units, rng, outcome)

    after = _totals(events, rows)
    outcome.realised = _relative_change(value_before, metric_value(after, metric))
    return outcome


def _move_rate(
    events: _Events,
    metric: str,
    rows: list[int],
    before: Totals,
    edit: Edit,
    units: float | None,
    rng: random.Random,
    outcome: _Outcome,
) -> None:
    """Flip flags on individual rows until the segment's numerator has moved.

    The denominator is never touched -- lowering fill rate does not delete requests, lowering
    CTR does not remove impressions -- so the rate moves in exact proportion to the counted
    numerator and the requested magnitude is delivered to within one row of rounding.
    """
    numerator_field = _FORMULA[metric][0]
    numerator = getattr(before, numerator_field)
    if units is None:
        wanted = int(round(numerator * (1.0 + edit.magnitude))) - numerator
    else:
        wanted = int(round(units))
    if wanted == 0:
        if not numerator and edit.magnitude > 0:
            outcome.notes.append(
                f"{edit.label}: the segment has no {numerator_field} to scale up, and a "
                "relative rise from zero has no defined size. Specify the move as a "
                "compensating pair against a segment that does, or pick another window."
            )
        return

    raising = wanted > 0
    eligible = _eligible(events, metric, rows, raising=raising)
    outcome.eligible = len(eligible)
    k = min(abs(wanted), len(eligible))

    # Uniform over the eligible rows, and that is load-bearing rather than a default. A
    # selection ordered by time, revenue or row position carries whatever those correlate with:
    # clearing the first N fills of a window moves the incident to the start of it, and
    # clearing the highest-revenue fills drags eCPM down alongside the fill rate, so the answer
    # key would name one metric while two had moved.
    if k:
        chosen = rng.sample(eligible, k)
        _mutate(events, metric, chosen, raising=raising, before=before, outcome=outcome)

    outcome.changed = k
    outcome.delta = float(k if raising else -k)
    if k < abs(wanted):
        delivered = k / numerator if numerator else 0.0
        outcome.notes.append(
            f"{edit.label}: asked to move {metric} by {edit.magnitude:+.1%} which needs "
            f"{abs(wanted):,} rows, but only {len(eligible):,} rows in the segment could be "
            f"flipped that way. Delivered {delivered:.1%} of the numerator instead."
        )


def _move_ecpm(
    events: _Events,
    rows: list[int],
    before: Totals,
    edit: Edit,
    units: float | None,
    outcome: _Outcome,
) -> None:
    """Scale revenue on impression rows.

    eCPM is revenue over impressions, so there is no flag to flip: the price moved, not the
    volume. Scaling every impression row by the same factor leaves impressions untouched, which
    means fill rate, render rate and CTR are all provably unchanged and the answer key names
    exactly one metric.
    """
    revenue = events.require("revenue")
    impression = events.require("is_impression")
    eligible = [i for i in rows if impression[i] and revenue[i]]
    outcome.eligible = len(eligible)
    if not eligible:
        outcome.notes.append(f"{edit.label}: no impression in this segment carried revenue.")
        return

    if units is None:
        factor = 1.0 + edit.magnitude
    else:
        factor = 1.0 + units / before.revenue if before.revenue else 1.0

    values = events.editable("revenue")
    for i in eligible:
        values[i] = values[i] * factor
    outcome.changed = len(eligible)
    outcome.delta = before.revenue * (factor - 1.0)


def _eligible(events: _Events, metric: str, rows: Iterable[int], *, raising: bool) -> list[int]:
    """Rows carrying a flag that can be flipped in the requested direction.

    Raising a rate is bounded by this in a way lowering it is not: there are always fills to
    clear, but a segment where 98% of fills already rendered has almost no room to raise render
    rate, and the honest response is to deliver what exists and say so.
    """
    filled = events.require("is_filled")
    impression = events.require("is_impression")
    click = events.require("is_click")

    if metric == "fill_rate":
        if raising:
            return [i for i in rows if not filled[i]]
        return [i for i in rows if filled[i]]
    if metric == "render_rate":
        if raising:
            return [i for i in rows if filled[i] and not impression[i]]
        return [i for i in rows if impression[i]]
    if metric == "ctr":
        if raising:
            return [i for i in rows if impression[i] and not click[i]]
        return [i for i in rows if click[i]]
    raise InjectionError(f"No row-level move defined for {metric!r}")


def _mutate(
    events: _Events,
    metric: str,
    chosen: Sequence[int],
    *,
    raising: bool,
    before: Totals,
    outcome: _Outcome,
) -> None:
    """Apply the flag change, cascading through the funnel in whichever direction it must.

    The cascade is here, in one place, rather than at each call site. A request that stops
    being filled cannot still have rendered, and a row that keeps an impression flag or a
    revenue figure after its fill is cleared is not a smaller incident, it is an event that
    could not exist -- and every rollup built from it would be internally consistent and wrong.

    Columns are opened for writing only where the cascade reaches them, so a CTR injection
    leaves the fill, impression and revenue columns as the originals rather than rewriting them
    with identical values.
    """
    click = events.editable("is_click")

    if not raising:
        if metric == "ctr":
            for i in chosen:
                click[i] = 0
            return
        impression = events.editable("is_impression")
        revenue = events.editable("revenue")
        filled = events.editable("is_filled") if metric == "fill_rate" else None
        for i in chosen:
            if filled is not None:
                filled[i] = 0
            impression[i] = 0
            revenue[i] = 0.0
            click[i] = 0
        return

    if metric == "ctr":
        for i in chosen:
            click[i] = 1
        return

    # Raising a rate creates rows further down the funnel that did not exist, and they have to
    # behave like their neighbours. Promoting a thousand requests to filled and leaving them
    # unrendered would drop the segment's render rate by exactly as much as its fill rate rose,
    # so the answer key would claim one metric moved while a detector correctly reported two --
    # scored as a false positive against an incident the injector itself created.
    stage_rate, click_rate, revenue_each = _prevailing(before, metric)
    downstream = int(round(len(chosen) * stage_rate))
    clicks = int(round(downstream * click_rate))
    impression = events.editable("is_impression")
    revenue = events.editable("revenue")

    if metric == "fill_rate":
        filled = events.editable("is_filled")
        for i in chosen:
            filled[i] = 1

    # Exact counts rather than a Bernoulli draw per row. A coin flip delivers the prevailing
    # rate only in expectation, and on the small segments these shapes are usually planted into
    # the sampling error is comparable to the incident being planted.
    for rank, i in enumerate(chosen):
        if rank >= downstream:
            break
        impression[i] = 1
        revenue[i] = revenue_each
        click[i] = 1 if rank < clicks else 0

    if metric == "fill_rate":
        advertiser = events.column("advertiser_id")
        if advertiser is not None and any(not advertiser[i] for i in chosen):
            outcome.notes.append(
                "Requests promoted to filled keep an empty advertiser_id, because inventing "
                "one would fabricate a foreign key. Those rows resolve to an empty vertical "
                "and campaign_type, so treat advertiser-sliced results on this run with care."
            )


def _prevailing(before: Totals, metric: str) -> tuple[float, float, float]:
    """The segment's own downstream behaviour, used to keep promoted rows unremarkable."""
    if metric == "fill_rate":
        stage_rate = before.impressions / before.fills if before.fills else 0.0
    else:
        stage_rate = 1.0
    click_rate = before.clicks / before.impressions if before.impressions else 0.0
    revenue_each = before.revenue / before.impressions if before.impressions else 0.0
    return stage_rate, click_rate, revenue_each


def _match_segment(
    events: _Events, where: tuple[tuple[str, str], ...], dimensions: DimensionIndex | None
) -> list[int]:
    """Row indices in a segment, ignoring time.

    A dimension is matched against a physical column when the event set carries one, and
    resolved through the dimension index otherwise. Neither being available is an error rather
    than an empty match: a typo in a dimension name would otherwise inject nothing at all and
    report a clean run with a rows_changed of zero, which reads exactly like a control.
    """
    if not where:
        return list(range(events.n))

    predicates: list[tuple[list[Any], str]] = []
    for name, value in where:
        column = events.column(name)
        if column is not None:
            predicates.append((column, value))
            continue
        lookup = (dimensions.lookups if dimensions else {}).get(name)
        if lookup is None:
            known = ", ".join((dimensions.dimensions if dimensions else []) or ["none"])
            raise InjectionError(
                f"Cannot resolve dimension {name!r}: the event set has no such column and the "
                f"dimension index does not define it (defines: {known}). Injecting nothing "
                "here would be indistinguishable from a clean control."
            )
        key_column, mapping = lookup
        keys = events.require(key_column)
        predicates.append(([mapping.get(k, "") for k in keys], value))

    first, wanted = predicates[0]
    rows = [i for i in range(events.n) if first[i] == wanted]
    for column, value in predicates[1:]:
        rows = [i for i in rows if column[i] == value]
    return rows


def _within(events: _Events, rows: Sequence[int], start_ms: int, end_ms: int) -> list[int]:
    """Restrict to the window, half-open so the slices of a drift ramp cannot double-count."""
    times = events.times
    return [i for i in rows if start_ms <= times[i] < end_ms]


def _totals(events: _Events, rows: Sequence[int]) -> Totals:
    filled = events.require("is_filled")
    impression = events.require("is_impression")
    click = events.require("is_click")
    revenue = events.require("revenue")
    fills = 0
    impressions = 0
    clicks = 0
    money = 0.0
    for i in rows:
        fills += filled[i]
        impressions += impression[i]
        clicks += click[i]
        money += revenue[i]
    return Totals(len(rows), fills, impressions, clicks, money)


def _count_violations(events: _Events) -> int:
    """Rows describing an event that could not have happened.

    Run over the whole output, not only the rows this injection touched. The cascade in
    ``_mutate`` is meant to make violations impossible, so this is an audit of that claim, and
    an audit restricted to the rows the author expected to be affected is not one.
    """
    filled = events.require("is_filled")
    impression = events.require("is_impression")
    click = events.require("is_click")
    revenue = events.require("revenue")
    bad = 0
    for f, i, c, r in zip(filled, impression, click, revenue, strict=True):
        if i > f or c > i or (not i and r):
            bad += 1
    return bad


def _probes(incident: IncidentPlan) -> list[tuple[str, tuple[tuple[str, str], ...], int, int]]:
    """Segments whose stillness is part of the claim, measured before and after.

    Only the two shapes that assert invisibility get probed. An interaction that moved a
    marginal is not an interaction, and a compensating pair that moved the grand total is not
    compensating; in both cases the shape silently becomes a main effect and the evaluation
    credits the detector with catching something it was never supposed to be able to see.
    """
    out: list[tuple[str, tuple[tuple[str, str], ...], int, int]] = []
    if not incident.edits:
        return out
    kind = incident.spec.kind
    if kind == KIND_INTERACTION:
        edit = incident.edits[0]
        for name, value in edit.where:
            out.append((f"{name}={value}", ((name, value),), edit.start_ms, edit.end_ms))
    elif kind == KIND_COMPENSATING_PAIR:
        edit = incident.edits[0]
        out.append(("__all__", (), edit.start_ms, edit.end_ms))
    return out


def _drift_edits(
    spec: IncidentSpec, where: tuple[tuple[str, str], ...], start_ms: int, end_ms: int
) -> tuple[Edit, ...]:
    """Slice the window into steps whose magnitudes ramp to the requested one.

    One step per day by default, because a daily rollup with a trailing weekly baseline is what
    this shape is designed to slip past, and a ramp finer than the baseline's own resolution
    would be testing the arithmetic rather than the blind spot.
    """
    days = (end_ms - start_ms) / 86_400_000
    steps = spec.steps or max(2, min(24, int(round(days))))
    if steps < 2:
        raise InjectionError(
            "A drift needs at least two steps; with one it is a step change and belongs under "
            "kind='main_effect'."
        )
    span = (end_ms - start_ms) / steps
    edits = []
    for n in range(steps):
        a = start_ms + int(round(n * span))
        b = start_ms + int(round((n + 1) * span))
        edits.append(Edit(where, a, b, spec.magnitude * (n + 1) / steps, f"drift{n + 1}"))
    return tuple(edits)


def _plan_clean(spec: IncidentSpec) -> IncidentPlan:
    if spec.magnitude != 0.0:
        raise InjectionError(
            f"A clean control was given a magnitude of {spec.magnitude}. The control exists to "
            "change nothing, and a control that moves a metric measures nothing at all."
        )
    return IncidentPlan(
        spec=spec,
        expected_segment={},
        expected_metric=spec.metric,
        expected_direction="none",
        rows_affected=0,
        description=(
            "Control run: no incident is injected and the data is returned unchanged. Every "
            "finding reported on this run is a false positive."
        ),
        expected_detectable=False,
        blind_spot=_BLIND_SPOTS[KIND_CLEAN],
        edits=(),
    )


def _reject_illegal_slice(metric: str, where: Mapping[str, str]) -> None:
    offending = sorted(set(where) & _POST_FILL_DIMENSIONS)
    if offending and metric in _REQUEST_GRAINED_METRICS:
        raise InjectionError(
            f"{metric} is measured over every request, but {', '.join(offending)} only exists "
            "on requests that were filled. Clearing a fill would remove the row from the "
            "segment entirely, so the segment's fill rate would stay at 1.0 and the answer key "
            "would record a drop that no query can reproduce."
        )


def _describe(spec: IncidentSpec, edits: tuple[Edit, ...], blind_spot: str) -> str:
    direction = "falls" if spec.magnitude < 0 else "rises"
    size = f"{abs(spec.magnitude):.0%}"
    segment = _label(spec.where)
    window = f"{_stamp(spec.start)} to {_stamp(spec.end)}"

    if spec.kind == KIND_INTERACTION:
        sentence = (
            f"{spec.metric} {direction} {size} only where {segment}, from {window}, leaving "
            "both one-way marginals flat."
        )
    elif spec.kind == KIND_COMPENSATING_PAIR:
        sentence = (
            f"{spec.metric} {direction} {size} for {segment} while "
            f"{_label(spec.counterpart or {})} moves the opposite way by the same counted "
            f"amount, from {window}, so the grand total does not move."
        )
    elif spec.kind == KIND_HIGH_CARDINALITY:
        sentence = (
            f"{spec.metric} {direction} {size} for the single entity {segment}, from {window}."
        )
    elif spec.kind == KIND_SLOW_DRIFT:
        sentence = (
            f"{spec.metric} drifts to {size} {'below' if spec.magnitude < 0 else 'above'} "
            f"baseline for {segment} across {len(edits)} steps, from {window}."
        )
    elif spec.kind == KIND_SPIKE:
        minutes = (spec.end - spec.start).total_seconds() / 60
        sentence = (
            f"{spec.metric} {direction} {size} for {segment} for {minutes:.0f} minutes from "
            f"{_stamp(spec.start)}."
        )
    else:
        sentence = f"{spec.metric} {direction} {size} for {segment}, from {window}."

    if blind_spot:
        sentence = f"{sentence} Expected to be missed: {blind_spot}"
    return sentence


def _label(where: Mapping[str, str]) -> str:
    return " and ".join(f"{k}={v}" for k, v in sorted(where.items())) or "all traffic"


def _freeze(where: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(where.items()))


def _relative_change(before: float | None, after: float | None) -> float:
    if before is None or after is None or not before:
        return 0.0
    return (after - before) / before


def _stamp(value: datetime) -> str:
    return _as_utc(value).strftime("%Y-%m-%d %H:%M UTC")


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _as_utc(value: datetime) -> datetime:
    """Naive timestamps are read as UTC.

    The parquet corpus stores timestamps without a zone and the DDL declares them UTC, so a
    spec written with an aware datetime and a corpus read without one would otherwise raise a
    TypeError on the first comparison -- or worse, be reconciled by assuming local time and
    place the whole window an hour away from the incident it was meant to describe.
    """
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _to_millis(value: Any) -> int:
    if isinstance(value, datetime):
        return int((_as_utc(value) - _EPOCH).total_seconds() * 1000)
    if isinstance(value, bool):
        raise InjectionError("event_time must be a datetime or epoch milliseconds, not a bool")
    if isinstance(value, (int, float)):
        return int(value)
    raise InjectionError(f"Cannot read {value!r} as a timestamp")


def _from_millis(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)


def _arrow_millis(column: Any) -> list[int]:
    """Timestamps as epoch milliseconds, without building nine million datetime objects.

    Comparing integers also sidesteps the aware/naive trap entirely: whatever zone the column
    declares, the epoch value is absolute.
    """
    if pa.types.is_timestamp(column.type):
        column = pc.cast(column, pa.timestamp("ms"), safe=False)
        return pc.cast(column, pa.int64()).to_pylist()
    if pa.types.is_integer(column.type):
        return column.to_pylist()
    return [_to_millis(v) for v in column.to_pylist()]
