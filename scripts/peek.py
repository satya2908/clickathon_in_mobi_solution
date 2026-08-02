"""Ad-hoc lattice peek. Prints a metric by one dimension for one day."""

from __future__ import annotations

import sys

from verdict.config import load_config
from verdict.db import ClickHouse


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-07-08"
    combos = sys.argv[2:] or ["publisher_tier", "os_version", "region"]

    ch = ClickHouse(load_config().clickhouse)
    for combo in combos:
        print(f"\n{combo} on {day}")
        rows = ch.query(
            """SELECT key_a,
                      round(sum(fills) / sum(requests), 4) AS fill_rate,
                      sum(requests) AS reqs
               FROM rollup_1d
               WHERE combo = {combo:String} AND bucket = {day:String}
               GROUP BY key_a ORDER BY fill_rate""",
            {"combo": combo, "day": day},
            name="peek",
        )
        for key, rate, requests in rows:
            print(f"  {key:<24} {rate:>8}  {requests:>10,}")


if __name__ == "__main__":
    main()
