"""Append one batch of events to a loaded corpus, the way a production tick would.

`verdict load` truncates and rebuilds, which is the right shape for the initial import and the
wrong shape for a batch arriving on a schedule. This inserts into `ad_events` and lets the
materialized views carry the new rows up through rollup_5m, rollup_1h and rollup_1d on the way
in, so "ingested" and "queryable at every grain" are the same instant. Nothing is truncated.

Timed end to end and reported per stage, because the interesting figure for a batch pipeline is
not how fast one insert is but how long a metric stays unanswerable after the file lands.

    python scripts/ingest_batch.py /path/to/ad_events.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdict.config import load_config  # noqa: E402
from verdict.db import ClickHouse  # noqa: E402
from verdict.load import FACT_COLUMNS, assert_not_lfs_stub  # noqa: E402


def main(path_str: str) -> int:
    path = Path(path_str)
    assert_not_lfs_stub(path)

    cfg = load_config()
    ch = ClickHouse(cfg.clickhouse)

    pf = pq.ParquetFile(path)
    rows, size = pf.metadata.num_rows, path.stat().st_size
    print(f"batch: {rows:,} events, {size / 1024 / 1024:.1f} MiB, "
          f"{pf.metadata.num_row_groups} row group(s)")

    before = ch.scalar("SELECT count() FROM ad_events", name="count_before", default=0)
    print(f"corpus before: {before:,} events")

    t0 = time.perf_counter()
    read_s = 0.0
    sent = 0
    for rg in range(pf.metadata.num_row_groups):
        r0 = time.perf_counter()
        table = pf.read_row_group(rg, columns=FACT_COLUMNS)
        read_s += time.perf_counter() - r0
        ch.insert_arrow("ad_events", table, name=f"append_rg{rg}")
        sent += table.num_rows
    insert_s = time.perf_counter() - t0

    # Not part of ingestion, but the only thing that makes the number meaningful: the rollups
    # the detectors actually read have to contain the batch before it can be called ingested.
    t1 = time.perf_counter()
    after = ch.scalar("SELECT count() FROM ad_events", name="count_after", default=0)
    lo, hi = ch.query(
        "SELECT min(event_time), max(event_time) FROM ad_events "
        "WHERE event_time >= (SELECT max(event_time) FROM ad_events) - INTERVAL 30 DAY",
        name="window",
    )[0][:2]
    checks = ch.query(
        """
        SELECT
          (SELECT count() FROM ad_events WHERE event_time >= {lo:DateTime}) AS raw,
          (SELECT sum(requests) FROM rollup_5m
             WHERE combo='__all__' AND bucket >= {lo:DateTime})            AS r5m,
          (SELECT sum(requests) FROM rollup_1h
             WHERE combo='__all__' AND bucket >= {lo:DateTime})            AS r1h,
          (SELECT sum(requests) FROM rollup_1d
             WHERE combo='__all__' AND bucket >= {lo:DateTime})            AS r1d
        """,
        {"lo": _batch_start(pf)},
        name="verify_rollups",
    )[0]
    verify_s = time.perf_counter() - t1

    print(f"corpus after:  {after:,} events  (+{after - before:,})")
    print(f"window now:    {lo} to {hi}")
    print()
    print(f"  read parquet   {read_s:8.2f}s")
    print(f"  insert + MVs   {insert_s - read_s:8.2f}s")
    print(f"  INGEST TOTAL   {insert_s:8.2f}s   ({rows / insert_s:,.0f} events/s)")
    print(f"  verify rollups {verify_s:8.2f}s")
    print()

    raw, r5m, r1h, r1d = (int(v) for v in checks)
    ok = raw == r5m == r1h == r1d
    print(f"  batch window: raw {raw:,} | 5m {r5m:,} | 1h {r1h:,} | 1d {r1d:,}")
    print("  rollups agree with raw" if ok else "  ROLLUPS DISAGREE WITH RAW")
    return 0 if ok else 1


def _batch_start(pf: pq.ParquetFile):
    import pyarrow.compute as pc

    return pc.min(pf.read(columns=["event_time"])["event_time"]).as_py()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
