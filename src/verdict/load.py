"""Dataset loading and post-load verification.

The verification is not ceremony. Two failure modes here are quiet enough to corrupt every
downstream conclusion while every command still reports success:

  * A dimension CSV that is really a Git LFS pointer stub. It parses as valid CSV with one
    strange row, the dictionary loads with three entries, dictGet returns '' for every lookup,
    and the whole lattice collapses into a single empty-string segment. Every metric still
    computes. Every answer is wrong.
  * Fact rows whose foreign keys miss the dimension tables. dictGet substitutes a default
    rather than failing, so unmatched traffic silently accumulates in an '' segment that looks
    like a real one.

Both are cheap to detect immediately and expensive to discover at hour eighteen.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from .db import ClickHouse
from .metrics import MetricRegistry
from .schema import backfill_statements

log = logging.getLogger(__name__)

_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

DIM_FILES = {
    "dim_apps": ("apps.csv", ["app_id", "category", "publisher_tier"]),
    "dim_advertisers": ("advertisers.csv", ["advertiser_id", "vertical", "campaign_type"]),
    "dim_geo_device": (
        "geo_device.csv",
        ["geo_device_id", "region", "country", "device_model", "os_version"],
    ),
}

FACT_FILE = "ad_events.parquet"
FACT_COLUMNS = [
    "event_time",
    "app_id",
    "geo_device_id",
    "advertiser_id",
    "ad_format",
    "is_filled",
    "is_impression",
    "is_click",
    "revenue",
]


class LoadError(RuntimeError):
    pass


@dataclass
class LoadReport:
    dim_rows: dict[str, int] = field(default_factory=dict)
    fact_rows: int = 0
    rollup_rows: dict[str, int] = field(default_factory=dict)
    orphans: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    window: tuple[str, str] = ("", "")
    warnings: list[str] = field(default_factory=list)


def assert_not_lfs_stub(path: Path) -> None:
    """Refuse to load a Git LFS pointer as if it were data."""
    with path.open("rb") as fh:
        head = fh.read(len(_LFS_MAGIC))
    if head == _LFS_MAGIC:
        raise LoadError(
            f"{path.name} is a Git LFS pointer stub, not the real file ({path.stat().st_size} "
            "bytes). Run `git lfs install && git lfs pull` in the dataset repository, or "
            "download the file directly. Loading it would produce a dictionary with no "
            "entries, and every dimension lookup would silently return an empty string."
        )


def load_dimensions(ch: ClickHouse, data_dir: Path, report: LoadReport) -> None:
    for table, (filename, columns) in DIM_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise LoadError(f"Missing dimension file {path}")
        assert_not_lfs_stub(path)

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in columns if c not in (reader.fieldnames or [])]
            if missing:
                raise LoadError(
                    f"{filename} is missing column(s) {missing}; found "
                    f"{reader.fieldnames}. Refusing to load a partial dimension."
                )
            rows = [[r[c] for c in columns] for r in reader]

        if not rows:
            raise LoadError(f"{filename} parsed to zero rows")

        ch.command(f"TRUNCATE TABLE IF EXISTS {table}", name=f"truncate_{table}")
        ch.insert(table, rows, columns, name=f"insert_{table}")
        report.dim_rows[table] = len(rows)
        log.info("loaded %s: %d rows", table, len(rows))

    reload_dictionaries(ch)
    verify_dictionaries(ch)


DICTIONARIES = {
    "dict_apps": ("dim_apps", "app_id", "category"),
    "dict_advertisers": ("dim_advertisers", "advertiser_id", "vertical"),
    "dict_geo_device": ("dim_geo_device", "geo_device_id", "region"),
}


def _cluster(ch: ClickHouse) -> str:
    """The cluster to broadcast to, or "" on a single-node service."""
    rows = ch.query(
        "SELECT cluster, count() AS n FROM system.clusters GROUP BY cluster ORDER BY n DESC",
        name="clusters",
    )
    for cluster, nodes in rows:
        if int(nodes) > 1:
            return str(cluster)
    return ""


def reload_dictionaries(ch: ClickHouse) -> None:
    """Reload every dimension dictionary on every node.

    `SYSTEM RELOAD DICTIONARY` is node-local, and the dictionaries are LIFETIME(0), so on a
    multi-node service the reload reaches whichever node answered the request and the others keep
    serving the copy they cached. Both paths that resolve a dimension -- the materialized view on
    insert and the backfill -- go through dictGet, so a batch inserted through a stale node is
    attributed with the previous dimension values and nothing anywhere reports an error.

    That is not hypothetical: the unseen dataset ships regenerated attributes for the same IDs,
    and its 1.5M events were rolled up entirely under the old ones because the insert landed on
    the node that had missed the reload. The rows looked fine, the totals reconciled against raw,
    and every segment in the five-day investigation named the wrong population.
    """
    cluster = _cluster(ch)
    on = f" ON CLUSTER {cluster!r}" if cluster else ""
    for dictionary in DICTIONARIES:
        # Qualified, because ON CLUSTER runs on the other nodes without this session's database.
        qualified = f"{ch.cfg.database}.{dictionary}"
        ch.command(f"SYSTEM RELOAD DICTIONARY {qualified}{on}", name=f"reload_{dictionary}")
    log.info("reloaded %d dictionaries%s", len(DICTIONARIES), f" on cluster {cluster}" if cluster else "")


def verify_dictionaries(ch: ClickHouse) -> None:
    """Confirm every node's dictionary agrees with the table it was built from.

    A dictionary that loaded zero rows still resolves every lookup, to the empty string, so the
    count and a round-trip are both checked. Neither catches a *stale* dictionary: it answers
    promptly, with a plausible value, that happens to be the previous load's. So the answer is
    compared against the dimension table, on every replica, which is the only check that can tell
    "loaded" from "loaded the right thing".
    """
    cluster = _cluster(ch)
    for name, (table, key, attr) in DICTIONARIES.items():
        loaded = ch.scalar(
            "SELECT element_count FROM system.dictionaries WHERE name = {n:String}",
            {"n": name},
            name=f"dict_count_{name}",
            default=0,
        )
        if not loaded:
            raise LoadError(f"Dictionary {name} reports zero elements after reload")

        sample, truth = ch.query(
            f"SELECT {key}, {attr} FROM {table} LIMIT 1", name=f"sample_{table}"
        )[0]

        source = f"clusterAllReplicas({cluster!r}, system.one)" if cluster else "system.one"
        qualified = f"{ch.cfg.database}.{name}"
        rows = ch.query(
            f"SELECT hostName(), dictGet('{qualified}', '{attr}', {{k:String}}) FROM {source}",
            {"k": sample},
            name=f"probe_{name}",
        )
        for node, resolved in rows:
            if not resolved:
                raise LoadError(
                    f"Dictionary {name} resolved {attr} to an empty string on {node} for a key "
                    f"that exists in {table} ({sample!r}). The lattice would be built on empty "
                    "segments."
                )
            if resolved != truth:
                raise LoadError(
                    f"Dictionary {name} is stale on {node}: {attr} for {sample!r} is "
                    f"{resolved!r} there but {truth!r} in {table}. Every segment built on that "
                    "node would name the wrong population."
                )
        log.info("dictionary %s: %s elements, %d node(s) agree", name, f"{loaded:,}", len(rows))


def load_facts(ch: ClickHouse, data_dir: Path, report: LoadReport, *, limit_rows: int | None = None) -> None:
    path = data_dir / FACT_FILE
    if not path.exists():
        raise LoadError(f"Missing fact file {path}")
    assert_not_lfs_stub(path)

    pf = pq.ParquetFile(path)
    schema_names = set(pf.schema_arrow.names)
    missing = [c for c in FACT_COLUMNS if c not in schema_names]
    if missing:
        raise LoadError(f"{FACT_FILE} is missing column(s) {missing}")

    ch.command("TRUNCATE TABLE IF EXISTS ad_events", name="truncate_ad_events")

    loaded = 0
    # Row groups are the natural insert unit: one is ~1M rows here, big enough that ClickHouse
    # writes a well-sized part and small enough to stay within memory.
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=FACT_COLUMNS)
        if limit_rows is not None and loaded + table.num_rows > limit_rows:
            table = table.slice(0, max(0, limit_rows - loaded))
        if table.num_rows == 0:
            break
        ch.insert_arrow("ad_events", table, name=f"insert_events_rg{rg}")
        loaded += table.num_rows
        log.info("loaded row group %d/%d (%s rows total)", rg + 1, pf.metadata.num_row_groups, f"{loaded:,}")
        if limit_rows is not None and loaded >= limit_rows:
            break

    report.fact_rows = loaded

    stored = ch.scalar("SELECT count() FROM ad_events", name="count_ad_events", default=0)
    if stored != loaded:
        raise LoadError(
            f"Inserted {loaded:,} rows but ad_events holds {stored:,}. Refusing to continue: "
            "every baseline would be computed on an unknown fraction of the data."
        )


def verify_integrity(ch: ClickHouse, report: LoadReport) -> None:
    """Count fact rows whose keys miss the dimension tables."""
    checks = [
        ("app_id", "dim_apps", "dict_apps", []),
        ("geo_device_id", "dim_geo_device", "dict_geo_device", []),
        # advertiser_id is legitimately empty on unfilled requests, so those rows are excluded
        # rather than counted as broken references.
        ("advertiser_id", "dim_advertisers", "dict_advertisers", ["advertiser_id != ''"]),
    ]
    for column, table, dictionary, extra in checks:
        conditions = [*extra, f"NOT dictHas('{dictionary}', {column})"]
        orphans = ch.scalar(
            f"SELECT count() FROM ad_events WHERE {' AND '.join(conditions)}",
            name=f"orphans_{column}",
            default=0,
        )
        report.orphans[column] = int(orphans)
        if orphans:
            report.warnings.append(
                f"{orphans:,} fact rows have {column} values absent from {table}. Their "
                f"dimension values resolve to '' and will appear as a distinct segment."
            )

    unfilled_with_advertiser = ch.scalar(
        "SELECT count() FROM ad_events WHERE is_filled = 0 AND advertiser_id != ''",
        name="unfilled_with_advertiser",
        default=0,
    )
    if unfilled_with_advertiser:
        report.warnings.append(
            f"{unfilled_with_advertiser:,} unfilled requests carry an advertiser_id, which "
            "contradicts the glossary. The post-fill dimension guard assumes they do not."
        )

    funnel_violations = ch.scalar(
        """SELECT count() FROM ad_events
           WHERE is_impression > is_filled OR is_click > is_impression
              OR (is_impression = 0 AND revenue > 0)""",
        name="funnel_violations",
        default=0,
    )
    if funnel_violations:
        report.warnings.append(
            f"{funnel_violations:,} rows violate the funnel ordering "
            "(request >= fill >= impression >= click, revenue only on impressions)."
        )


def backfill_rollups(ch: ClickHouse, registry: MetricRegistry, report: LoadReport) -> None:
    """Populate rollups from loaded history.

    Materialized views only fire on new inserts, so history has to be pushed through the same
    transformation explicitly. It is literally the same SELECT text, generated once, which is
    what keeps the backfilled past and the streamed future on one definition.
    """
    for table in ("rollup_5m", "rollup_1h", "rollup_1d"):
        ch.command(f"TRUNCATE TABLE IF EXISTS {table}", name=f"truncate_{table}")

    for stmt in backfill_statements(registry.lattice_dimensions):
        log.info("running %s", stmt.name)
        ch.command(stmt.sql, name=stmt.name)

    for table in ("rollup_5m", "rollup_1h", "rollup_1d"):
        # Best-effort. Immediately after a bulk load the background merge pool is saturated and
        # ClickHouse Cloud refuses new OPTIMIZE work with CANNOT_ASSIGN_OPTIMIZE. That is a
        # scheduling state, not a data problem: the rows are present and correct either way,
        # merges continue on their own, and every read sums rather than assuming one row per
        # key. The row counts below are reported as-is so the log shows the pre-merge reality
        # rather than implying a compaction that did not happen.
        merged = ch.try_command(f"OPTIMIZE TABLE {table} FINAL", name=f"optimize_{table}")
        if not merged:
            report.warnings.append(
                f"{table} was not compacted: the merge pool was busy. Row counts below are "
                f"pre-merge and will fall as ClickHouse merges in the background. No total "
                f"changes, because reads aggregate explicitly."
            )
        report.rollup_rows[table] = int(
            ch.scalar(f"SELECT count() FROM {table}", name=f"count_{table}", default=0)
        )


def verify_rollups(ch: ClickHouse, registry: MetricRegistry, report: LoadReport) -> None:
    """Assert every rollup grain agrees with the raw events, exactly.

    This is the single most valuable check in the loader. If a rollup disagrees with the facts,
    every number in every case file is wrong in a way no amount of statistical care can catch,
    because the statistics would be internally consistent.
    """
    raw = ch.query(
        """SELECT count(), sum(is_filled), sum(is_impression), sum(is_click), sum(revenue)
           FROM ad_events""",
        name="raw_totals",
    )[0]

    for table in ("rollup_5m", "rollup_1h", "rollup_1d"):
        got = ch.query(
            f"""SELECT sum(requests), sum(fills), sum(impressions), sum(clicks), sum(revenue)
                FROM {table} WHERE combo = '__all__'""",
            name=f"rollup_totals_{table}",
        )[0]
        for i, label in enumerate(["requests", "fills", "impressions", "clicks", "revenue"]):
            expected, actual = raw[i], got[i]
            if label == "revenue":
                # Float summation order differs between the raw scan and the rollup chain, so
                # compare relatively rather than demanding bit equality.
                if expected and abs(actual - expected) / abs(expected) > 1e-9:
                    raise LoadError(
                        f"{table} revenue total {actual!r} disagrees with ad_events {expected!r}"
                    )
            elif int(actual or 0) != int(expected or 0):
                raise LoadError(
                    f"{table} {label} total is {actual:,} but ad_events has {expected:,}. "
                    "The rollup and the facts disagree; every downstream number is unreliable."
                )

    # Each one-way combo must also reconstruct the grand total on its own. A combo that fails
    # this has a missing or duplicated key rather than a wrong overall sum.
    for dim in registry.lattice_dimensions:
        total = ch.scalar(
            "SELECT sum(requests) FROM rollup_1d WHERE combo = {c:String}",
            {"c": dim},
            name=f"combo_total_{dim}",
            default=0,
        )
        if int(total or 0) != int(raw[0]):
            raise LoadError(
                f"Combo {dim!r} sums to {int(total or 0):,} requests but the grand total is "
                f"{int(raw[0]):,}. The lattice is not a partition of the data."
            )

    report.metrics = golden_metrics(ch, registry)
    window = ch.query(
        "SELECT min(event_time), max(event_time) FROM ad_events", name="event_window"
    )[0]
    report.window = (str(window[0]), str(window[1]))


def golden_metrics(ch: ClickHouse, registry: MetricRegistry) -> dict[str, float]:
    """Compute every metric two ways -- from raw events and from the rollup -- and require
    agreement. This is what makes the glossary formulas testable rather than aspirational."""
    out: dict[str, float] = {}
    for name, metric in registry.metrics.items():
        raw_value = ch.scalar(
            f"SELECT {metric.value_sql(from_rollup=False)} FROM ad_events",
            name=f"golden_raw_{name}",
        )
        rollup_value = ch.scalar(
            f"SELECT {metric.value_sql(from_rollup=True)} FROM rollup_1d WHERE combo = '__all__'",
            name=f"golden_rollup_{name}",
        )
        if raw_value is None or rollup_value is None:
            raise LoadError(f"Metric {name} evaluated to NULL over the whole dataset")
        if abs(float(raw_value) - float(rollup_value)) > 1e-6 * max(1.0, abs(float(raw_value))):
            raise LoadError(
                f"Metric {name} is {raw_value!r} from raw events but {rollup_value!r} from the "
                "rollup. The stored counters do not reproduce the glossary formula."
            )
        out[name] = float(raw_value)
    return out


def load_all(
    ch: ClickHouse,
    registry: MetricRegistry,
    data_dir: str | Path,
    *,
    limit_rows: int | None = None,
) -> LoadReport:
    data = Path(data_dir)
    if not data.is_dir():
        raise LoadError(f"Data directory {data} does not exist")

    report = LoadReport()
    load_dimensions(ch, data, report)
    load_facts(ch, data, report, limit_rows=limit_rows)
    verify_integrity(ch, report)
    backfill_rollups(ch, registry, report)
    verify_rollups(ch, registry, report)
    return report


@dataclass
class IngestReport:
    """What one batch cost, stage by stage."""

    events: int = 0
    file_bytes: int = 0
    corpus_before: int = 0
    corpus_after: int = 0
    dimensions_refreshed: bool = False
    refresh_s: float = 0.0
    read_s: float = 0.0
    insert_s: float = 0.0
    verify_s: float = 0.0
    batch_window: tuple[datetime, datetime] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ingest_s(self) -> float:
        """Arrival to queryable: everything that has to happen before a detector can read it."""
        return self.refresh_s + self.read_s + self.insert_s


def dimensions_changed(ch: ClickHouse, data_dir: Path) -> bool:
    """Whether the CSVs in this folder say something different from the loaded dimension tables.

    Compared by value rather than by presence, because a release that reissues the dimension
    files unchanged is the common case and rebuilding the whole lattice for it would turn a
    twenty-second ingest into a three-minute one.
    """
    for table, (filename, columns) in DIM_FILES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        assert_not_lfs_stub(path)
        with path.open(newline="") as fh:
            incoming = sorted(tuple(r[c] for c in columns) for r in csv.DictReader(fh))
        loaded = sorted(
            tuple(str(v) for v in row)
            for row in ch.query(f"SELECT {', '.join(columns)} FROM {table}", name=f"dump_{table}")
        )
        if incoming != loaded:
            log.info("%s differs from %s; dimensions need refreshing", filename, table)
            return True
    return False


def ingest(ch: ClickHouse, registry: MetricRegistry, path: str | Path) -> IngestReport:
    """Add one batch of events to a corpus that is already loaded, and leave it queryable.

    Nothing is truncated. The materialized views carry the new rows up through rollup_5m,
    rollup_1h and rollup_1d as they go in, so "ingested" and "readable at every grain" are the
    same instant -- which is the only definition of ingest latency that means anything to a
    detector waiting on the data.

    If the folder also carries dimension CSVs and they differ from what is loaded, the rollups
    over the existing events are re-derived first. Skipping that would leave history describing
    segments by their old attributes and the new batch describing them by their new ones, and
    every baseline would then compare two different populations wearing one label.
    """
    p = Path(path)
    if p.is_dir():
        data_dir, fact = p, p / FACT_FILE
    else:
        data_dir, fact = p.parent, p
    if not fact.exists():
        raise LoadError(f"No events file at {fact}")
    assert_not_lfs_stub(fact)

    report = IngestReport(file_bytes=fact.stat().st_size)
    pf = pq.ParquetFile(fact)
    report.events = pf.metadata.num_rows
    missing = [c for c in FACT_COLUMNS if c not in set(pf.schema_arrow.names)]
    if missing:
        raise LoadError(f"{fact.name} is missing column(s) {missing}")

    report.corpus_before = int(
        ch.scalar("SELECT count() FROM ad_events", name="count_before", default=0)
    )

    if dimensions_changed(ch, data_dir):
        start = time.perf_counter()
        refreshed = refresh_dimensions(ch, registry, data_dir)
        report.refresh_s = time.perf_counter() - start
        report.dimensions_refreshed = True
        report.warnings.extend(refreshed.warnings)
    else:
        # Still forced on every node: the views resolve dimensions through dictGet as the rows
        # go in, and a node serving a dictionary from a previous load attributes the whole batch
        # to the wrong segments without raising anything.
        reload_dictionaries(ch)
        verify_dictionaries(ch)

    for rg in range(pf.metadata.num_row_groups):
        start = time.perf_counter()
        table = pf.read_row_group(rg, columns=FACT_COLUMNS)
        report.read_s += time.perf_counter() - start
        start = time.perf_counter()
        ch.insert_arrow("ad_events", table, name=f"ingest_rg{rg}")
        report.insert_s += time.perf_counter() - start

    start = time.perf_counter()
    report.corpus_after = int(
        ch.scalar("SELECT count() FROM ad_events", name="count_after", default=0)
    )
    if report.corpus_after - report.corpus_before != report.events:
        raise LoadError(
            f"Inserted {report.events:,} rows but the corpus grew by "
            f"{report.corpus_after - report.corpus_before:,}. Refusing to report a batch as "
            "ingested when the count disagrees."
        )
    report.batch_window = _verify_batch(ch, fact, report.events)
    report.verify_s = time.perf_counter() - start
    return report


def _verify_batch(ch: ClickHouse, fact: Path, events: int) -> tuple[datetime, datetime]:
    """Reconcile every rollup grain against raw over the batch's own window.

    The global check in `verify_rollups` scans the whole corpus and is too slow to run per batch;
    this asks the same question about the rows that just arrived.
    """
    import pyarrow.compute as pc

    times = pq.read_table(fact, columns=["event_time"])["event_time"]
    lo, hi = pc.min(times).as_py(), pc.max(times).as_py()

    raw, r5m, r1h, r1d = ch.query(
        """
        SELECT
          (SELECT count() FROM ad_events
             WHERE event_time BETWEEN {lo:DateTime} AND {hi:DateTime})            AS raw,
          (SELECT sum(requests) FROM rollup_5m
             WHERE combo='__all__' AND bucket BETWEEN {lo:DateTime} AND {hi:DateTime}) AS r5m,
          (SELECT sum(requests) FROM rollup_1h
             WHERE combo='__all__' AND bucket BETWEEN {lo:DateTime} AND {hi:DateTime}) AS r1h,
          (SELECT sum(requests) FROM rollup_1d
             WHERE combo='__all__' AND bucket BETWEEN {lo:DateTime} AND {hi:DateTime}) AS r1d
        """,
        {"lo": lo, "hi": hi},
        name="verify_batch",
    )[0]

    grains = {"rollup_5m": int(r5m), "rollup_1h": int(r1h), "rollup_1d": int(r1d)}
    disagree = {name: n for name, n in grains.items() if n != int(raw)}
    if disagree:
        raise LoadError(
            f"Over the batch window {lo} to {hi}, ad_events holds {int(raw):,} rows but "
            + ", ".join(f"{name} holds {n:,}" for name, n in disagree.items())
            + ". The batch is not queryable at every grain."
        )
    log.info("batch verified: %s rows agree across raw and all three grains", f"{events:,}")
    return lo, hi


def refresh_dimensions(ch: ClickHouse, registry: MetricRegistry, data_dir: str | Path) -> LoadReport:
    """Swap the dimension tables and rebuild the rollups over the events already loaded.

    The rollups denormalise `region`, `os_version` and the rest as the rows go in, so new
    dimension values do not reach data that is already stored: every existing rollup row keeps
    encoding the previous meaning of its segment, and comparing across the swap compares two
    different populations wearing the same label.

    Reloading the whole corpus fixes that and is mostly wasted work, because `ad_events` is
    unchanged -- only the projection of it is stale. This re-derives the rollups from the facts
    already in ClickHouse, which costs a scan rather than a re-upload.
    """
    data = Path(data_dir)
    if not data.is_dir():
        raise LoadError(f"Data directory {data} does not exist")

    report = LoadReport()
    load_dimensions(ch, data, report)

    report.fact_rows = int(ch.scalar("SELECT count() FROM ad_events", name="count_facts", default=0))
    if not report.fact_rows:
        raise LoadError(
            "ad_events is empty, so there is nothing to re-derive. Use `verdict load` to import "
            "a corpus first."
        )

    verify_integrity(ch, report)
    backfill_rollups(ch, registry, report)
    verify_rollups(ch, registry, report)
    return report
