"""Chooses what to plant, so that scoring measures generalisation rather than memory.

Accuracy against the incidents already in a corpus measures how well code fits data its
author has read. It is the easiest number in the project to make look good and the least
informative one, because every threshold, every dimension in the lattice and every choice
about which tests to run was made by someone who had already seen those six movements.

This module exists to produce incidents nobody designed against. The rule it follows
everywhere: **segments are chosen by rank in the data, never by name.** Not one product,
country or version string appears below. Writing ``os_version=Android 15`` here would smuggle
the development corpus back into the evaluation through the side door, and the sweep would
stop transferring to any other dataset the moment the hardware changed.

The catalogue deliberately includes shapes the design is known to handle badly. Each carries
``expected_detectable``, so a miss on a known boundary scores as a boundary and an unexpected
miss stands out as a real regression. A catalogue of only winnable incidents would be a
second way of measuring the same thing the development corpus already measures.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from .inject import KINDS, MOVABLE_METRICS, DimensionIndex, IncidentSpec, InjectionError

# Where in the corpus to plant. Anchored on the first event rather than on a date, because a
# fixed date is a fact about one dataset, and offset by enough days for a baseline to exist:
# every temporal expectation is built from history preceding its window, so an incident planted
# on day one would be scored against a detector that had nothing to compare it with. That is a
# statement about the detector's inputs, not about which parts of this corpus happen to be calm.
_HISTORY_DAYS = 7
_WINDOW_HOURS = 24

# Which segment to move, expressed as a position in the traffic distribution rather than a
# name. Mid-sized is the interesting case in both directions -- the largest segment moves the
# global metric so far that any detector finds it, and the smallest cannot clear a denominator
# floor no matter how hard it is pushed, so neither tells you much about the system.
_MAIN_RANK = 0.35
_PAIR_RANK = 0.25
_DRIFT_RANK = 0.30

# Magnitudes. Chosen against the system's own declared reporting floor of 5%, not against the
# sizes of the incidents in the development corpus: one comfortably above it, one close enough
# to it to be a genuine test of sensitivity.
_LARGE = -0.35
_MODEST = -0.12


def _ranked_values(
    table: Any, dims: DimensionIndex, dimension: str, *, minimum_share: float = 0.02
) -> list[tuple[str, int]]:
    """Every value of one dimension, ordered by the traffic it carries, largest first.

    Values below ``minimum_share`` of total traffic are dropped. Planting in a segment too
    small to resolve the move would produce an answer key the data cannot support, and the
    system would be scored as missing something that was never measurable -- which is a
    statement about arithmetic, not about the detector.
    """
    if dimension not in dims.lookups:
        raise InjectionError(f"No dimension {dimension!r} in the index; have {dims.dimensions}")
    key_column, mapping = dims.lookups[dimension]
    if key_column not in table.column_names:
        raise InjectionError(f"Fact table has no column {key_column!r} to resolve {dimension!r}")

    counts: Counter[str] = Counter()
    for key in table.column(key_column).to_pylist():
        value = mapping.get(key)
        if value:
            counts[value] += 1

    total = sum(counts.values())
    if not total:
        raise InjectionError(f"Dimension {dimension!r} resolved to nothing across the corpus")
    floor = total * minimum_share
    return [(v, n) for v, n in counts.most_common() if n >= floor]


def _ranked_column(table: Any, column: str, *, minimum_share: float = 0.0005) -> list[tuple[str, int]]:
    """The same ordering for a dimension the fact table carries directly.

    Needed for the high-cardinality shape, whose whole point is a key that lives outside the
    rollup lattice. The share floor is far lower than for a lattice dimension because an
    individual app is a fraction of a percent of traffic by definition -- that is what makes it
    high-cardinality -- but it is not zero, since a key with a handful of rows would produce an
    answer key no amount of detection could recover.
    """
    if column not in table.column_names:
        raise InjectionError(f"Fact table has no column {column!r}")
    counts = Counter(v for v in table.column(column).to_pylist() if v)
    total = sum(counts.values())
    if not total:
        raise InjectionError(f"Column {column!r} is empty across the corpus")
    floor = max(total * minimum_share, 500)
    return [(v, n) for v, n in counts.most_common() if n >= floor]


def _pick(values: list[tuple[str, int]], rank: float, offset: int = 0) -> str:
    """Take the value sitting at a fractional position in the traffic ordering."""
    if not values:
        raise InjectionError("No dimension value carries enough traffic to plant an incident in")
    index = (int(rank * len(values)) + offset) % len(values)
    return values[index][0]


def _windows(table: Any) -> tuple[datetime, datetime]:
    column = "event_time"
    if column not in table.column_names:
        raise InjectionError(f"Fact table has no {column!r} column")
    times = table.column(column).to_pylist()
    first, last = min(times), max(times)
    if isinstance(first, int):  # epoch millis
        first = datetime.utcfromtimestamp(first / 1000.0)
        last = datetime.utcfromtimestamp(last / 1000.0)
    anchor = first.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=_HISTORY_DAYS)
    if anchor >= last:
        raise InjectionError(
            f"The corpus spans {first} to {last}, which leaves no room for a window after "
            f"{_HISTORY_DAYS} days of baseline history."
        )
    return anchor, anchor + timedelta(hours=_WINDOW_HOURS)


def built_in_sweep(table: Any, dims: DimensionIndex, *, seed: int = 0) -> list[IncidentSpec]:
    """Seven incidents spanning what the system should catch and what it is known to miss.

    ``seed`` shifts every segment choice by the same offset through the traffic ordering, so a
    second run plants the same shapes somewhere else. That is the cheapest available guard
    against tuning to a particular sweep: if the score moves a lot between seeds, the system is
    fitting segments rather than detecting movements.

    Each incident gets its own window, spaced so that one cannot be read as the baseline for
    the next. They are placed in the closing days of the corpus because that is where a
    streaming system would be looking, and the choice is deliberately made without reference
    to which parts of this particular corpus happen to be quiet.
    """
    start, end = _windows(table)

    def slot(n: int) -> tuple[datetime, datetime]:
        shift = timedelta(hours=_WINDOW_HOURS * n)
        return start + shift, end + shift

    def values(dimension: str) -> list[tuple[str, int]]:
        """Rank a dimension wherever it lives, rather than assuming which side it is on.

        Some lattice dimensions are columns on the fact table and some are dictionary lookups,
        and which is which is a property of the schema. An earlier version quietly fell back to
        a different dimension when a name was not in the index, and the compensating pair was
        planted on a country value under an ad_format key -- matching nothing, injecting
        nothing, and reporting a healthy zero-row incident that scored as a detector miss.
        """
        if dimension in dims.lookups:
            return _ranked_values(table, dims, dimension)
        if dimension in table.column_names:
            return _ranked_column(table, dimension, minimum_share=0.02)
        raise InjectionError(
            f"Dimension {dimension!r} is neither a fact column {table.column_names} nor an "
            f"indexed lookup {dims.dimensions}"
        )

    geo = values("country")
    fmt = values("ad_format")
    tier = values("publisher_tier")
    device = values("device_model")

    s0, e0 = slot(0)
    s1, e1 = slot(1)
    s2, e2 = slot(2)
    s3, e3 = slot(3)
    s4, e4 = slot(4)
    s5, e5 = slot(5)
    s6, e6 = slot(6)

    country = _pick(geo, _MAIN_RANK, seed)
    fmt_a = _pick(fmt, _PAIR_RANK, seed)
    fmt_b = _pick(fmt, _PAIR_RANK, seed + 1)
    return [
        # Should be caught: a plain one-dimensional collapse, the shape every detector exists
        # for. If this is missed nothing else in the sweep is worth reading.
        IncidentSpec(
            kind="main_effect", metric="fill_rate", where={"country": country},
            start=s0, end=e0, magnitude=_LARGE, seed=seed,
            note="one-dimensional collapse in a mid-sized segment",
        ),
        # Should be caught, and is the reason the structural detector exists: invisible in both
        # one-way marginals, visible only in the cell. On fill rate rather than click-through
        # because a two-dimensional cell of clicks is a handful of rows: clicks are a couple of
        # percent of impressions, so intersecting two dimensions on them plants an incident too
        # small for any denominator floor to admit, and the sweep would be measuring arithmetic
        # rather than detection.
        IncidentSpec(
            kind="interaction", metric="fill_rate",
            where={"publisher_tier": _pick(tier, 0.0, seed),
                   "device_model": _pick(device, 0.0, seed)},
            start=s1, end=e1, magnitude=_LARGE, seed=seed,
            note="two-dimensional cell, both marginals left flat",
        ),
        # Should be caught, and tests the half of the pair a drop-only detector never sees.
        IncidentSpec(
            kind="compensating_pair", metric="ecpm", where={"ad_format": fmt_a},
            counterpart={"ad_format": fmt_b},
            start=s2, end=e2, magnitude=_LARGE, seed=seed,
            note="two opposing moves leaving the global metric flat",
        ),
        # Sensitivity, not shape: close enough to the 5% reporting floor to be a real question
        # about whether the denominator supports the claim. The largest segment and a doubled
        # window, because this one has to fail for the right reason -- too small an effect to
        # resolve -- rather than because clicks are sparse and the cell never had the traffic.
        IncidentSpec(
            kind="main_effect", metric="ctr", where={"country": _pick(geo, 0.0, seed)},
            start=s3, end=e3 + timedelta(hours=_WINDOW_HOURS), magnitude=_MODEST, seed=seed,
            note="modest move, tests the floor rather than the shape",
        ),
        # Expected miss: dimensions marked in_lattice: false have no rollup cell at any grain,
        # so there is nothing for a detector to test. A property of the storage shape.
        IncidentSpec(
            kind="high_cardinality", metric="fill_rate",
            where={"app_id": _pick(_ranked_column(table, "app_id"), _MAIN_RANK, seed)},
            start=s4, end=e4, magnitude=_LARGE, seed=seed,
            note="single rogue key outside the lattice",
        ),
        # Expected miss, and the most interesting of the three: a decline spread over days is
        # partly absorbed into the trailing baseline it would have to be measured against.
        IncidentSpec(
            kind="slow_drift", metric="ecpm",
            where={"publisher_tier": _pick(tier, 0.0, seed)},
            start=s5, end=e5 + timedelta(days=3), magnitude=_LARGE, seed=seed,
            note="gradual decline, absorbed into its own baseline",
        ),
        # The control. Correct output is silence; anything reported here is a false positive,
        # and without it the sweep would only ever measure recall.
        IncidentSpec(
            kind="clean", metric="fill_rate", where={},
            start=s6, end=e6, magnitude=0.0, seed=seed,
            note="nothing planted; any finding is a false positive",
        ),
    ]


def specs_from_json(text: str) -> list[IncidentSpec]:
    """Parse a hand-written catalogue, for reproducing one incident or exploring a new shape."""
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise InjectionError("A catalogue must be a JSON list of incident objects")
    return [_spec(entry, i) for i, entry in enumerate(raw)]


def _spec(entry: dict[str, Any], index: int) -> IncidentSpec:
    missing = {"kind", "metric", "start", "end"} - set(entry)
    if missing:
        raise InjectionError(f"Incident {index} is missing {sorted(missing)}")
    if entry["kind"] not in KINDS:
        raise InjectionError(f"Incident {index} has unknown kind {entry['kind']!r}; expected {list(KINDS)}")
    if entry["metric"] not in MOVABLE_METRICS:
        raise InjectionError(
            f"Incident {index} names metric {entry['metric']!r}, which the injector cannot move. "
            f"Movable: {list(MOVABLE_METRICS)}"
        )
    return IncidentSpec(
        kind=entry["kind"],
        metric=entry["metric"],
        where=dict(entry.get("where") or {}),
        start=datetime.fromisoformat(entry["start"]),
        end=datetime.fromisoformat(entry["end"]),
        magnitude=float(entry.get("magnitude", 0.0)),
        seed=int(entry.get("seed", 0)),
        note=str(entry.get("note", "")),
        counterpart=dict(entry["counterpart"]) if entry.get("counterpart") else None,
        steps=int(entry.get("steps", 0)),
    )


__all__ = ["built_in_sweep", "specs_from_json"]
