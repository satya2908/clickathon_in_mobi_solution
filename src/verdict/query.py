"""Reading counters out of the rollup lattice.

The central type is ``Counters``: an additive tuple of requests, fills, impressions, clicks,
and revenue. Because counters subtract cleanly and metrics are divided out afterwards, the
explain-away test -- remove the accused segment and see whether the parent metric returns to
normal -- is literally ``total - candidate``. No bespoke SQL per candidate, and no chance of a
counterfactual that quietly uses a different denominator than the observation it is compared
against.

This is also why nothing stores a metric. Fill rates cannot be subtracted from one another.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .db import ClickHouse
from .metrics import Metric
from .schema import GRAINS, TOTAL_COMBO

_GRAIN_DELTA = {"5m": timedelta(minutes=5), "1h": timedelta(hours=1), "1d": timedelta(days=1)}


@dataclass(frozen=True)
class Window:
    """A half-open time range ``[start, end)`` at a given grain."""

    start: datetime
    end: datetime
    grain: str = "1h"

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Window end {self.end} is not after start {self.start}")
        if self.grain not in GRAINS:
            raise ValueError(f"Unknown grain {self.grain!r}; expected one of {list(GRAINS)}")

    @property
    def table(self) -> str:
        return GRAINS[self.grain][0]

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def buckets(self) -> int:
        return max(1, int(self.duration / _GRAIN_DELTA[self.grain]))

    def shifted(self, weeks: int) -> Window:
        """The same window some whole number of weeks earlier.

        Shifting by whole weeks is what makes the comparison like-for-like on both cycles at
        once: the same weekday and the same hours of day, so a weekend never looks like an
        incident and a nightly trough never looks like an outage.
        """
        delta = timedelta(weeks=weeks)
        return replace(self, start=self.start - delta, end=self.end - delta)

    def label(self) -> str:
        return f"{self.start:%Y-%m-%d %H:%M} to {self.end:%Y-%m-%d %H:%M}"


@dataclass(frozen=True)
class Segment:
    """A conjunction of dimension equalities, e.g. region=APAC AND os_version=iOS 18.1."""

    keys: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def total() -> Segment:
        return Segment(())

    @staticmethod
    def of(**kwargs: str) -> Segment:
        return Segment(tuple(sorted(kwargs.items())))

    @property
    def depth(self) -> int:
        return len(self.keys)

    @property
    def is_total(self) -> bool:
        return not self.keys

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.keys)

    @property
    def combo(self) -> str:
        """The lattice combo holding this segment. Dimension names are sorted so that
        ``region|os_version`` and ``os_version|region`` cannot both exist."""
        if self.is_total:
            return TOTAL_COMBO
        return "|".join(sorted(self.dimensions))

    @property
    def combo_keys(self) -> tuple[str, str]:
        if self.is_total:
            return ("", "")
        ordered = sorted(self.keys)
        if len(ordered) == 1:
            return (ordered[0][1], "")
        return (ordered[0][1], ordered[1][1])

    def parent(self, drop: str) -> Segment:
        """This segment with one dimension removed."""
        return Segment(tuple((k, v) for k, v in self.keys if k != drop))

    def with_key(self, dimension: str, value: str) -> Segment:
        return Segment(tuple(sorted({**dict(self.keys), dimension: value}.items())))

    def label(self) -> str:
        if self.is_total:
            return "all traffic"
        return " AND ".join(f"{k}={v}" for k, v in sorted(self.keys))

    def as_dict(self) -> dict[str, str]:
        return dict(self.keys)

    def __str__(self) -> str:
        return self.label()


@dataclass(frozen=True)
class Counters:
    requests: int = 0
    fills: int = 0
    impressions: int = 0
    clicks: int = 0
    revenue: float = 0.0

    def __add__(self, other: Counters) -> Counters:
        return Counters(
            self.requests + other.requests,
            self.fills + other.fills,
            self.impressions + other.impressions,
            self.clicks + other.clicks,
            self.revenue + other.revenue,
        )

    def __sub__(self, other: Counters) -> Counters:
        """Counterfactual removal of a segment.

        Results are clamped at zero. A negative counter can only arise from asking for a
        counterfactual the lattice cannot express, and a negative request count propagating
        into a rate would produce a plausible-looking number with no meaning at all.
        """
        return Counters(
            max(0, self.requests - other.requests),
            max(0, self.fills - other.fills),
            max(0, self.impressions - other.impressions),
            max(0, self.clicks - other.clicks),
            max(0.0, self.revenue - other.revenue),
        )

    @property
    def empty(self) -> bool:
        return self.requests == 0

    def numerator(self, metric: Metric) -> float:
        return float(getattr(self, metric.numerator_field))

    def denominator(self, metric: Metric) -> float | None:
        field = metric.denominator_field
        return float(getattr(self, field)) if field else None

    def value(self, metric: Metric) -> float | None:
        """Evaluate a metric, mirroring ``Metric.value_sql`` exactly.

        Returns None rather than zero for an empty denominator. A segment with no impressions
        has no CTR; it does not have a CTR of zero, and the difference decides whether a
        significance test fires.
        """
        num = self.numerator(metric)
        den = self.denominator(metric)
        if den is None:
            return num
        if den == 0:
            return None
        return (num / den) * metric.scale


COUNTER_COLUMNS = ("requests", "fills", "impressions", "clicks", "revenue")
_SUMS = ", ".join(f"sum({c}) AS {c}" for c in COUNTER_COLUMNS)


def _row_to_counters(row: tuple) -> Counters:
    return Counters(int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]))


class RollupReader:
    """Every read of the rollup goes through here, so window semantics are defined once."""

    def __init__(self, ch: ClickHouse) -> None:
        self.ch = ch

    def total(self, window: Window) -> Counters:
        rows = self.ch.query(
            f"""SELECT {_SUMS} FROM {window.table}
                WHERE combo = {{combo:String}} AND bucket >= {{s:DateTime}} AND bucket < {{e:DateTime}}""",
            {"combo": TOTAL_COMBO, "s": window.start, "e": window.end},
            name="rollup_total",
        )
        return _row_to_counters(rows[0]) if rows and rows[0][0] is not None else Counters()

    def segment(self, segment: Segment, window: Window) -> Counters:
        if segment.is_total:
            return self.total(window)
        key_a, key_b = segment.combo_keys
        rows = self.ch.query(
            f"""SELECT {_SUMS} FROM {window.table}
                WHERE combo = {{combo:String}} AND key_a = {{a:String}} AND key_b = {{b:String}}
                  AND bucket >= {{s:DateTime}} AND bucket < {{e:DateTime}}""",
            {
                "combo": segment.combo,
                "a": key_a,
                "b": key_b,
                "s": window.start,
                "e": window.end,
            },
            name="rollup_segment",
        )
        return _row_to_counters(rows[0]) if rows and rows[0][0] is not None else Counters()

    def slice(self, combo: str, window: Window) -> dict[Segment, Counters]:
        """Every occupied cell of one combo over one window."""
        dims = combo.split("|") if combo != TOTAL_COMBO else []
        rows = self.ch.query(
            f"""SELECT key_a, key_b, {_SUMS} FROM {window.table}
                WHERE combo = {{combo:String}} AND bucket >= {{s:DateTime}} AND bucket < {{e:DateTime}}
                GROUP BY key_a, key_b""",
            {"combo": combo, "s": window.start, "e": window.end},
            name="rollup_slice",
        )
        out: dict[Segment, Counters] = {}
        for row in rows:
            if not dims:
                seg = Segment.total()
            elif len(dims) == 1:
                seg = Segment(((dims[0], row[0]),))
            else:
                seg = Segment(tuple(sorted(((dims[0], row[0]), (dims[1], row[1])))))
            out[seg] = _row_to_counters(row[2:])
        return out

    def slice_with_history(
        self, combo: str, window: Window, weeks: int
    ) -> dict[Segment, list[Counters]]:
        """One combo's cells for the window and each of the preceding ``weeks`` weeks.

        Index 0 is the window under investigation; 1..weeks are the aligned historical
        comparators, oldest last. Done in one query rather than ``weeks + 1`` so that all arms
        are read from a single consistent snapshot of the table -- a merge landing between two
        separate reads would shift the baseline underneath the observation.
        """
        dims = combo.split("|") if combo != TOTAL_COMBO else []
        rows = self.ch.query(
            f"""SELECT w, key_a, key_b, {_SUMS}
                FROM (
                    SELECT bucket, key_a, key_b, {', '.join(COUNTER_COLUMNS)}
                    FROM {window.table}
                    WHERE combo = {{combo:String}}
                      AND bucket >= {{hist_start:DateTime}} AND bucket < {{e:DateTime}}
                )
                ARRAY JOIN range(0, {{k:UInt8}} + 1) AS w
                WHERE bucket >= {{s:DateTime}} - toIntervalWeek(w)
                  AND bucket <  {{e:DateTime}} - toIntervalWeek(w)
                GROUP BY w, key_a, key_b""",
            {
                "combo": combo,
                "s": window.start,
                "e": window.end,
                "hist_start": window.shifted(weeks).start,
                "k": weeks,
            },
            name="rollup_slice_history",
        )

        out: dict[Segment, list[Counters]] = {}
        for row in rows:
            w = int(row[0])
            if not dims:
                seg = Segment.total()
            elif len(dims) == 1:
                seg = Segment(((dims[0], row[1]),))
            else:
                seg = Segment(tuple(sorted(((dims[0], row[1]), (dims[1], row[2])))))
            slot = out.setdefault(seg, [Counters() for _ in range(weeks + 1)])
            slot[w] = _row_to_counters(row[3:])
        return out

    def series(self, segment: Segment, start: datetime, end: datetime, grain: str) -> list[tuple[datetime, Counters]]:
        """A time series for one segment, for charting and onset detection."""
        table = GRAINS[grain][0]
        if segment.is_total:
            combo, key_a, key_b = TOTAL_COMBO, "", ""
        else:
            combo = segment.combo
            key_a, key_b = segment.combo_keys
        rows = self.ch.query(
            f"""SELECT bucket, {_SUMS} FROM {table}
                WHERE combo = {{combo:String}} AND key_a = {{a:String}} AND key_b = {{b:String}}
                  AND bucket >= {{s:DateTime}} AND bucket < {{e:DateTime}}
                GROUP BY bucket ORDER BY bucket""",
            {"combo": combo, "a": key_a, "b": key_b, "s": start, "e": end},
            name="rollup_series",
        )
        return [(row[0], _row_to_counters(row[1:])) for row in rows]

    def data_bounds(self) -> tuple[datetime, datetime] | None:
        rows = self.ch.query(
            f"SELECT min(bucket), max(bucket) FROM rollup_1d WHERE combo = '{TOTAL_COMBO}'",
            name="data_bounds",
        )
        if not rows or rows[0][0] is None:
            return None
        return (rows[0][0], rows[0][1])
