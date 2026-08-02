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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .db import ClickHouse
from .metrics import Metric
from .schema import GRAINS, LATTICE_DEPTH, TOTAL_COMBO

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
        """Counterfactual removal of a segment, refusing rather than clamping.

        A negative counter means the subtrahend was not contained in the minuend, so the
        counterfactual is not one the lattice can express. That happens for real: two
        expectations averaged over different sets of baseline weeks are not nested even when
        the segments are.

        Treating the fields as independent is worse than useless here, because they are the
        numerators and denominators of each other. (1000 req, 800 fills) - (900 req, 100 fills)
        leaves (100, 700), in which no field is negative and the fill rate is 700%: impossible,
        silently wrong, and precise enough to look deliberate. So the test is not
        non-negativity but funnel coherence -- see `coherent`.

        An incoherent result collapses to empty, which `value` reports as None for any rate
        metric, and every caller already handles None by declining to draw a conclusion.
        """
        out = Counters(
            self.requests - other.requests,
            self.fills - other.fills,
            self.impressions - other.impressions,
            self.clicks - other.clicks,
            self.revenue - other.revenue,
        )
        return out if out.coherent else Counters()

    @property
    def coherent(self) -> bool:
        """Whether these counters describe a population that could exist.

        Every event is a request; a filled request may yield one impression; an impression may
        yield one click. So requests >= fills >= impressions >= clicks holds in any real
        population, and every rate in config/metrics.yaml is a ratio across adjacent stages of
        it. A tuple violating the ordering has no interpretation -- it is not an unusual
        segment, it is not a segment at all.
        """
        return (
            self.requests >= self.fills >= self.impressions >= self.clicks >= 0
            and self.revenue >= 0.0
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


class ComboNotStored(LookupError):
    """A combo was requested from a grain that does not materialize it."""


def _require_stored(combo: str, window: Window) -> list[str]:
    """Split a combo into its dimensions, refusing one the grain does not carry.

    Grains store different lattice depths: five-minute rollups carry one-way cells only, while
    hourly and daily carry pairs. A query for a pair against the five-minute table is perfectly
    valid SQL and returns nothing, which every caller reads as "no such segment moved" rather
    than "this was never looked at". A detector reports a clean grid it never saw; a localizer
    drops the children that would have refuted its accusation. Both are silent, and both are
    wrong in the direction of unwarranted confidence, so this raises instead.
    """
    if combo == TOTAL_COMBO:
        return []
    dims = combo.split("|")
    stored = LATTICE_DEPTH.get(window.grain, 2)
    if len(dims) > stored:
        raise ComboNotStored(
            f"combo {combo!r} is {len(dims)} deep but grain {window.grain!r} materializes "
            f"depth {stored}. Query a coarser grain, or raise LATTICE_DEPTH[{window.grain!r}] "
            f"and reload."
        )
    return dims


def subtract(minuend: Counters, subtrahend: Counters) -> Counters | None:
    """Counterfactual removal that says so when it cannot be done.

    ``Counters.__sub__`` collapses an inexpressible removal to empty, which `value` reports as
    None for every rate metric. Revenue has no denominator, so the same collapse reads as a
    genuine zero instead. Call this wherever the difference between "the remainder is zero" and
    "there is no remainder to speak of" changes a verdict.
    """
    out = Counters(
        minuend.requests - subtrahend.requests,
        minuend.fills - subtrahend.fills,
        minuend.impressions - subtrahend.impressions,
        minuend.clicks - subtrahend.clicks,
        minuend.revenue - subtrahend.revenue,
    )
    return out if out.coherent else None


COUNTER_COLUMNS = ("requests", "fills", "impressions", "clicks", "revenue")
_SUMS = ", ".join(f"sum({c}) AS {c}" for c in COUNTER_COLUMNS)


def _row_to_counters(row: tuple) -> Counters:
    return Counters(int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]))


class RollupReader:
    """Every read of the rollup goes through here, so window semantics are defined once.

    ``read_mode`` decides how the lattice is fetched, and on a remote service it is the single
    largest term in how long a run takes.

    ``per_combo`` issues one query per combination, which is the obvious shape and the wrong
    one twice over. The counters it selects -- requests, fills, impressions, clicks, revenue --
    do not depend on which metric is being scanned, so ten metrics over forty-six combinations
    ask for the same forty-six results ten times. And each of those queries is around ten
    milliseconds of ClickHouse behind a round trip several times longer, so the run spends most
    of its life waiting rather than computing.

    ``batch`` reads the whole lattice for the window and its baseline weeks in one query and
    answers every later call from memory. Same rows, same arithmetic, same verdicts; the
    difference is entirely in how many times the network is crossed to get them.

    Both are kept because the comparison is worth being able to run, and because a single
    result set is a memory cost that a much larger lattice might eventually not want to pay.
    """

    def __init__(self, ch: ClickHouse, *, read_mode: str = "batch") -> None:
        self.ch = ch
        self.read_mode = read_mode
        # (combo, window, weeks) -> cells. Populated by prefetch_lattice, read by
        # slice_with_history and slice. Keyed by window because a holdout half and the full
        # window are different questions that would otherwise collide.
        self._lattice: dict[tuple[str, Window, int], dict[Segment, list[Counters]]] = {}

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

    def prefetch_lattice(self, combos: Sequence[str], window: Window, weeks: int) -> int:
        """Read the whole lattice, window and baseline arms together, in one query.

        No-op under ``per_combo``. Returns the number of cells cached, which the caller logs so
        the batch is visible in the trace rather than being an invisible speedup.

        ``combo`` is a column, so asking for forty-six of them costs one ``IN`` and one more
        grouping key. The result is the same rows the per-combo path would fetch one at a time.
        Combos the grain does not materialize are skipped rather than raising: this is a
        prefetch, and a caller that genuinely needs one still gets the honest refusal from
        ``_require_stored`` when it asks.
        """
        if self.read_mode != "batch":
            return 0

        wanted = [c for c in dict.fromkeys(combos) if (c, window, weeks) not in self._lattice]
        stored: dict[str, list[str]] = {}
        for combo in wanted:
            try:
                stored[combo] = _require_stored(combo, window)
            except ComboNotStored:
                continue
        if not stored:
            return 0

        rows = self.ch.query(
            f"""SELECT combo, w, key_a, key_b, {_SUMS}
                FROM (
                    SELECT bucket, combo, key_a, key_b, {', '.join(COUNTER_COLUMNS)}
                    FROM {window.table}
                    WHERE combo IN {{combos:Array(String)}}
                      AND bucket >= {{hist_start:DateTime}} AND bucket < {{e:DateTime}}
                )
                ARRAY JOIN range(0, {{k:UInt8}} + 1) AS w
                WHERE bucket >= {{s:DateTime}} - toIntervalWeek(w)
                  AND bucket <  {{e:DateTime}} - toIntervalWeek(w)
                GROUP BY combo, w, key_a, key_b""",
            {
                "combos": list(stored),
                "s": window.start,
                "e": window.end,
                "hist_start": window.shifted(weeks).start,
                "k": weeks,
            },
            name="rollup_lattice_batch",
        )

        # Every requested combo gets an entry even if it came back empty, so a later call
        # reads "nothing there" from the cache instead of going to the network to be told the
        # same thing.
        cells: dict[str, dict[Segment, list[Counters]]] = {c: {} for c in stored}
        for row in rows:
            combo, w = row[0], int(row[1])
            dims = stored[combo]
            if not dims:
                seg = Segment.total()
            elif len(dims) == 1:
                seg = Segment(((dims[0], row[2]),))
            else:
                seg = Segment(tuple(sorted(((dims[0], row[2]), (dims[1], row[3])))))
            slot = cells[combo].setdefault(seg, [Counters() for _ in range(weeks + 1)])
            slot[w] = _row_to_counters(row[4:])

        for combo, found in cells.items():
            self._lattice[(combo, window, weeks)] = found
        return sum(len(v) for v in cells.values())

    def segment_with_history(
        self, segment: Segment, window: Window, weeks: int
    ) -> list[Counters]:
        """One segment over the window and each aligned baseline week, index 0 the window.

        The same shape ``slice_with_history`` returns for a whole combo, for a single cell.
        Served from a prefetched lattice when one covers this window, which is what turns the
        holdout -- two half-windows times five arms for every candidate tested -- from a few
        hundred round trips into a read from memory.
        """
        cached = self._lattice.get((segment.combo, window, weeks))
        if cached is not None:
            return cached.get(segment, [Counters() for _ in range(weeks + 1)])
        return [self.segment(segment, window.shifted(w)) for w in range(weeks + 1)]

    def slice(self, combo: str, window: Window) -> dict[Segment, Counters]:
        """Every occupied cell of one combo over one window."""
        dims = _require_stored(combo, window)

        # Arm 0 of a prefetched lattice is exactly this window, so the structural detector
        # rides on the batch the temporal one already paid for.
        for (cached_combo, cached_window, _), cells in self._lattice.items():
            if cached_combo == combo and cached_window == window:
                return {seg: arms[0] for seg, arms in cells.items()}

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
        dims = _require_stored(combo, window)

        cached = self._lattice.get((combo, window, weeks))
        if cached is not None:
            return cached

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
