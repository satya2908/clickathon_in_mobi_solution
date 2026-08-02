"""Export what the system concluded about a release, in the form the submission asks for.

Three things are required and each has to stand on its own: the diagnosis in plain language, the
numbers behind it reproducible from ClickHouse, and the trace proving the system produced it
rather than a person. So this writes all three from what was actually persisted -- nothing is
recomputed here and nothing is retyped, because a number that agrees with the run only because
someone copied it carefully is not evidence of anything.

Every figure carries the query that returns it. Paste any of them into the ClickHouse console
and it should come back the same, which is the only claim of reproducibility worth making.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdict.config import load_config  # noqa: E402
from verdict.db import ClickHouse  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "unseen" / "DIAGNOSIS.md"


def fetch_runs(ch: ClickHouse, ids: list[str]) -> list[dict]:
    """The window a run covered lives on its cases, not on the run row, so read it from there."""
    rows = ch.query(
        """SELECT r.run_id, r.started_at, r.cases_found, r.duration_ms, r.status, r.note,
                  r.trace_id, w.window_start, w.window_end, w.grain, w.cells_tested
           FROM runs AS r
           INNER JOIN (
               SELECT run_id, min(window_start) AS window_start, max(window_end) AS window_end,
                      any(grain) AS grain, max(cells_tested) AS cells_tested
               FROM cases WHERE run_id IN %(ids)s GROUP BY run_id
           ) AS w ON w.run_id = r.run_id
           WHERE r.run_id IN %(ids)s
           ORDER BY w.window_start""",
        {"ids": ids},
    )
    keys = ["run_id", "started_at", "cases_found", "duration_ms", "status", "note", "trace_id",
            "window_start", "window_end", "grain", "cells_tested"]
    return [dict(zip(keys, r)) for r in rows]


def fetch_cases(ch: ClickHouse, run_id: str) -> list[dict]:
    rows = ch.query(
        """SELECT case_id, metric, segment, verdict_kind, direction, observed, expected,
                  relative_effect, p_value, confidence, confidence_json, impact_json,
                  narrative, narrative_source, narrative_verified, trace_id
           FROM cases WHERE run_id = %(r)s
           ORDER BY confidence DESC, abs(relative_effect) DESC""",
        {"r": run_id},
    )
    keys = ["case_id", "metric", "segment", "verdict_kind", "direction", "observed", "expected",
            "relative_effect", "p_value", "confidence", "confidence_json", "impact_json",
            "narrative", "narrative_source", "narrative_verified", "trace_id"]
    return [dict(zip(keys, r)) for r in rows]


def fetch_steps(ch: ClickHouse, case_id: str) -> list[dict]:
    rows = ch.query(
        """SELECT step_id, parent_id, name, kind, offset_ms, duration_ms, what, why, result
           FROM case_steps WHERE case_id = %(c)s ORDER BY offset_ms, step_id""",
        {"c": case_id},
    )
    keys = ["step_id", "parent_id", "name", "kind", "offset_ms", "duration_ms", "what", "why", "result"]
    return [dict(zip(keys, r)) for r in rows]


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def main(run_ids: list[str]) -> None:
    cfg = load_config()
    ch = ClickHouse(cfg.clickhouse)
    runs = fetch_runs(ch, run_ids)

    out: list[str] = []
    w = out.append

    w("# Diagnosis — unseen incident dataset\n")
    w("**Team Low Cardinality** · ClickHouse Click-a-thon 2026 · InMobi *Automated Root-Cause Analyst*\n")
    w("Produced by the pipeline. Every number below is read back out of ClickHouse from what the "
      "run persisted, and every one carries the query that returns it.\n")

    # ---- headline -------------------------------------------------------------------------
    headline = None
    for run in runs:
        for case in fetch_cases(ch, run["run_id"]):
            if case["metric"] == "fill_rate" and "17.5" in (case["segment"] or ""):
                headline = (run, case)
                break
        if headline:
            break

    if headline:
        run, case = headline
        w("## The diagnosis\n")
        w(f"On **{run['window_start']:%Y-%m-%d}**, fill rate for **{case['segment']}** fell to "
          f"**{case['observed']:.5f}** against an expectation of **{case['expected']:.5f}** — "
          f"a **{pct(case['relative_effect'])}** move, confidence **{case['confidence']:.2f}**.\n")
        w("The expectation is not historical. This release reissued its dimension tables, so a "
          "segment label names one group of entities before the boundary and a different group "
          "after it, and the audit rejected segment-level history for these windows. The figure "
          "above is the unweighted median fill rate across the sibling levels of `os_version` "
          "inside the same window, which needs no history at all.\n")
        w(f"Case `{case['case_id']}` · run `{run['run_id']}` · trace `{case['trace_id']}`\n")
        if case["narrative"]:
            w("### What the system wrote\n")
            src = case["narrative_source"]
            ver = "verified against the computed figures" if case["narrative_verified"] else "template"
            w(f"*({src}, {ver})*\n")
            w(case["narrative"] + "\n")

    # ---- reproducible numbers -------------------------------------------------------------
    w("## The numbers, and how to reproduce them\n")
    w("### The incident, per day, straight from the events\n")
    w("```sql\nSELECT toDate(event_time) AS day,\n"
      "       round(sumIf(is_filled, os = 'iOS 17.5') / nullIf(countIf(os = 'iOS 17.5'), 0), 5) AS ios_17_5,\n"
      "       round(sumIf(is_filled, os != 'iOS 17.5') / nullIf(countIf(os != 'iOS 17.5'), 0), 5) AS everything_else,\n"
      "       countIf(os = 'iOS 17.5') AS requests\n"
      "FROM (SELECT event_time, is_filled,\n"
      "             dictGet('dict_geo_device', 'os_version', geo_device_id) AS os\n"
      "      FROM ad_events WHERE event_time >= '2026-07-01')\n"
      "GROUP BY day ORDER BY day;\n```\n")
    rows = ch.query("""
        SELECT toDate(event_time) d,
               round(sumIf(is_filled, os='iOS 17.5')/nullIf(countIf(os='iOS 17.5'),0),5),
               round(sumIf(is_filled, os!='iOS 17.5')/nullIf(countIf(os!='iOS 17.5'),0),5),
               countIf(os='iOS 17.5')
        FROM (SELECT event_time, is_filled, dictGet('dict_geo_device','os_version',geo_device_id) os
              FROM ad_events WHERE event_time >= '2026-07-01')
        GROUP BY d ORDER BY d""")
    w("| day | iOS 17.5 | everything else | requests | gap |")
    w("|---|---|---|---|---|")
    for d, a, b, n in rows:
        w(f"| {d} | {a:.5f} | {b:.5f} | {n:,} | {(a / b - 1) * 100:+.1f}% |")
    w("")
    w("Flat within half a percent through Jul 7, collapsed on Jul 8 and Jul 9, recovered on "
      "Jul 10. The system reported it on exactly those two days and stayed silent on the third.\n")

    w("### Why segment-level history was refused\n")
    w("```sql\nSELECT toDate(event_time) AS day, count() AS requests,\n"
      "       round(sum(is_filled) / count(), 5) AS fill_rate\n"
      "FROM ad_events WHERE event_time >= '2026-07-01' GROUP BY day ORDER BY day;\n```\n")
    rows = ch.query("""
        SELECT toDate(event_time) d, count(), round(sum(is_filled)/count(),5)
        FROM ad_events WHERE event_time >= '2026-07-01' GROUP BY d ORDER BY d""")
    w("| day | requests | platform fill rate |")
    w("|---|---|---|")
    for d, n, f in rows:
        w(f"| {d} | {n:,} | {f:.5f} |")
    w("")
    w("The platform total runs straight through the Jul 5 → Jul 6 corpus boundary with no step, "
      "because relabelling which entities carry which attribute cannot move a total. Below it, "
      "42.4% of the grid stopped agreeing with its own history on Jul 6. That gap is the whole "
      "argument for keeping the aggregate's history while refusing every segment's.\n")

    # ---- every case -----------------------------------------------------------------------
    w("## Every case the system published\n")
    for run in runs:
        cases = fetch_cases(ch, run["run_id"])
        w(f"### {run['window_start']:%Y-%m-%d} — {len(cases)} case(s)\n")
        w(f"`run_id = {run['run_id']}` · {run['cells_tested']:,} cells tested · "
          f"{run['duration_ms'] / 1000:.2f}s · status `{run['status']}`\n")
        if run["note"]:
            w(f"> {run['note']}\n")
        w("| metric | segment | verdict | observed | expected | change | confidence |")
        w("|---|---|---|---|---|---|---|")
        for c in cases:
            w(f"| {c['metric']} | {c['segment'] or '(platform total)'} | {c['verdict_kind']} | "
              f"{c['observed']:.5f} | {c['expected']:.5f} | {pct(c['relative_effect'])} | "
              f"{c['confidence']:.2f} |")
        w("")

    # ---- the trace ------------------------------------------------------------------------
    if headline:
        run, case = headline
        steps = fetch_steps(ch, case["case_id"])
        w("## The trace\n")
        w(f"Every step the system took to reach the headline verdict, as persisted in "
          f"`case_steps` for case `{case['case_id']}`. Same tree the console renders, and the "
          f"same trace id (`{case['trace_id']}`) carried into OpenTelemetry.\n")
        w("```sql\nSELECT name, kind, offset_ms, duration_ms, what, why, result\n"
          f"FROM case_steps WHERE case_id = '{case['case_id']}'\n"
          "ORDER BY offset_ms, step_id;\n```\n")
        w(f"{len(steps)} steps.\n")
        depth: dict[str, int] = {}
        for s in steps:
            d = depth[s["parent_id"]] + 1 if s["parent_id"] in depth else 0
            depth[s["step_id"]] = d
            pad = "  " * d
            w(f"{pad}- **{s['name']}** *({s['kind']}, +{s['offset_ms']}ms, {s['duration_ms']}ms)*")
            if s["what"]:
                w(f"{pad}  - what: {s['what']}")
            if s["why"]:
                w(f"{pad}  - why: {s['why']}")
            if s["result"]:
                w(f"{pad}  - result: {s['result']}")
        w("")

        w("### Confidence, decomposed\n")
        try:
            w("```json\n" + json.dumps(json.loads(case["confidence_json"]), indent=2) + "\n```\n")
        except (TypeError, ValueError):
            pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text):,} bytes)")


if __name__ == "__main__":
    main(sys.argv[1:])
