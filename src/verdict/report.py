"""Renders an investigation as one self-contained HTML file.

Self-contained is the requirement that shapes everything else here. No CDN, no build step, no
server: the output is a single file that opens from a filesystem, survives being emailed, and
renders identically in two years when whatever chart library was fashionable today has changed
its API twice. Everything the page needs -- data, styles, behaviour -- is inlined.

The structure follows the shape of the argument rather than the shape of the data. A verdict
is only worth as much as the reasoning a reader can audit, so the case file is built around
making that reasoning walkable: every case is a chain of steps, every step says what was done,
why it was done and what came back, and every number on the page traces to a computation. The
right-hand panel exists so that clicking a step answers those three questions without the
reader having to hold the previous screen in their head.

Nothing here computes anything. If a figure is not already on the case it does not appear on
the page -- a report that derived its own numbers could disagree with the database it claims
to be reporting, and the disagreement would be invisible.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .detect import CoverageGap, Finding
    from .localize import Candidate, Check
    from .pipeline import InvestigationResult
    from .store import Case

# Verdict states get colours; everything else is neutral. Three states rather than two,
# because a check that could not run is not a check that failed, and collapsing them would
# make the page assert something the system deliberately refuses to assert.
_STATE_COLOUR = {
    "pass": "ok",
    "fail": "bad",
    "unknown": "unknown",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float | None, places: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{places}f}%"


def _sig(value: float | None, places: int = 4) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0"
    return f"{value:.{places}g}"


def _rate(value: float | None, finding: Finding) -> str:
    """Render a metric level the way the metric is normally read."""
    if value is None:
        return "—"
    if finding.metric in {"fill_rate", "render_rate", "ctr"}:
        return f"{value * 100:.2f}%"
    return f"{value:,.4g}"


def _check_step(check: Check) -> dict[str, str]:
    """One localization test, as a walkable step.

    The ``why`` strings are the only prose in this module that is not read off the case, and
    they are here rather than in the statistics code because they explain the *purpose* of a
    test, which does not vary by run. The ``result`` always comes from the check's own detail
    string, written next to the arithmetic that produced it.
    """
    why = {
        "sufficiency": (
            "A segment can move a great deal and still be innocent: when a parent metric falls, "
            "every segment correlated with the cause falls with it. This asks the only question "
            "that separates a cause from a passenger -- if this segment is removed from the "
            "population, does the parent metric come back?"
        ),
        "minimality": (
            "An answer can be true and still too broad. If removing one child of this segment "
            "accounts for nearly all of the deviation, then the real cause is that child, and "
            "naming the parent sends someone looking in the wrong place."
        ),
        "maximality": (
            "The mirror of minimality. If this segment's siblings all moved with it, the cause "
            "is not specific to this segment at all and the honest answer is the level above."
        ),
        "holdout": (
            "A segment chosen because it deviated over a window had every chance to be chosen "
            "by noise. Splitting the window and checking the effect reproduces on the half it "
            "was not selected on is the cheapest guard against that."
        ),
    }
    titles = {
        "sufficiency": "Sufficiency — does removing it restore the parent?",
        "minimality": "Minimality — is it narrow enough?",
        "maximality": "Maximality — is it broad enough?",
        "holdout": "Holdout — does the effect reproduce?",
    }
    what = {
        "sufficiency": (
            "Recomputed the parent metric with this segment's traffic subtracted, and compared "
            "the result against the parent's original deviation."
        ),
        "minimality": (
            "Found the child of this segment carrying the most of the deviation, removed it, "
            "and measured how much deviation remained."
        ),
        "maximality": (
            "Measured how many of this segment's siblings, within the same dimension, moved in "
            "the same direction over the same window."
        ),
        "holdout": (
            "Split the window in half, measured the effect independently in each, and compared "
            "the two."
        ),
    }
    return {
        "id": check.name,
        "title": titles.get(check.name, check.name.title()),
        "state": check.state,
        "score": "—" if check.score is None else f"{check.score:.3f}",
        "what": what.get(check.name, ""),
        "why": why.get(check.name, ""),
        "result": check.detail or "No detail recorded.",
    }


def _steps_for(case: Case) -> list[dict[str, str]]:
    """The investigation as an ordered chain, from detection to verdict."""
    finding = case.finding
    loc = case.localization
    accused = loc.accused
    test = finding.test
    steps: list[dict[str, str]] = []

    detector_why = {
        "temporal": (
            "Compares each cell against its own past, aligned to the same weekday and hour, so "
            "that a Tuesday morning is judged against previous Tuesday mornings rather than "
            "against yesterday evening. This is what catches a segment that simply dropped."
        ),
        "structural": (
            "Compares each cell against what the rest of the grid implies it should be, using "
            "no history at all. This is what catches an interaction: a cell that is wrong "
            "relative to its own row and column while both of those look perfectly normal, "
            "which a comparison against the past can miss entirely when the totals stay flat."
        ),
    }
    steps.append(
        {
            "id": "detect",
            "title": f"Detection — {finding.detector}",
            "state": "pass",
            "score": _sig(test.p_value, 3),
            "what": (
                f"Tested {_esc(finding.segment.label())} on {finding.metric} over "
                f"{finding.window.label()}. Observed {_rate(test.observed, finding)} against an "
                f"expected {_rate(test.expected, finding)}, a relative move of "
                f"{_pct(finding.relative_effect)}."
            ),
            "why": detector_why.get(finding.detector, ""),
            "result": (
                f"z = {_sig(test.z, 4)}, p = {_sig(test.p_value, 3)}, model {test.model}. "
                f"Variance inflated by an overdispersion factor of {finding.phi:.2f}, estimated "
                f"from history excluding this window so that the anomaly cannot widen the very "
                f"interval used to judge it. Baseline pooled over {finding.weeks_kept} of "
                f"{finding.weeks_seen} aligned weeks."
                + (
                    ""
                    if finding.resolvable_effect is None
                    else (
                        f" On the traffic this cell carried, the smallest move it could have "
                        f"resolved is {_pct(finding.resolvable_effect)}, against a reporting "
                        f"threshold of {_pct(finding.effect_threshold, 0)}."
                    )
                )
            ),
        }
    )

    steps.append(
        {
            "id": "correct",
            "title": "Multiple-testing correction",
            "state": "pass" if finding.survives_correction else "fail",
            "score": "",
            "what": (
                "Applied a Benjamini-Hochberg correction across every cell tested in this run, "
                "not just the ones that looked interesting."
            ),
            "why": (
                "Thousands of cells are tested per run. At a five percent threshold, pure noise "
                "produces dozens of significant results, so an uncorrected p-value here means "
                "very little. The correction has to see the p-values that failed as well as "
                "the ones that passed, because the number of tests is its entire input."
            ),
            "result": (
                "Survived the correction."
                if finding.survives_correction
                else "Did not survive the correction and would not be reported on its own."
            ),
        }
    )

    mode_result = (
        f"Parent {loc.parent.label()} moved {_pct(loc.parent_deviation)}, "
        f"from {_rate(loc.parent_expected, finding)} to {_rate(loc.parent_observed, finding)}."
    )
    steps.append(
        {
            "id": "mode",
            "title": f"Strategy — {loc.mode.replace('_', ' ')}",
            "state": "pass",
            "score": f"{len(loc.candidates)} candidates",
            "what": (
                f"Enumerated {len(loc.candidates)} candidate segments and chose how to judge "
                "them, based on whether the parent metric moved at all."
            ),
            "why": (
                "The counterfactual test only means something if there is a parent deviation to "
                "restore. When the parent is flat -- which is what a compensating pair looks "
                "like globally -- removing a candidate proves nothing, so candidates have to be "
                "judged on their own movement and on whether their siblings share it instead."
            ),
            "result": mode_result + (f" {loc.note}" if loc.note else ""),
        }
    )

    if accused is not None:
        for name in ("sufficiency", "minimality", "maximality", "holdout"):
            check = accused.checks.get(name)
            if check is not None:
                steps.append(_check_step(check))

        cleared = loc.cleared
        if cleared:
            lines = "; ".join(
                f"{c.segment.label()} predicted {_rate(c.predicted_if_innocent, finding)}, "
                f"read {_rate(c.observed_value, finding)}"
                for c in cleared[:6]
            )
            steps.append(
                {
                    "id": "exonerate",
                    "title": f"Exoneration ledger — {len(cleared)} cleared",
                    "state": "pass",
                    "score": str(len(cleared)),
                    "what": (
                        "For each candidate that was not accused, predicted what it should read "
                        "if the accused segment explains everything, then compared that "
                        "prediction against what it actually read."
                    ),
                    "why": (
                        "Clearing a segment by not mentioning it is not evidence of anything. "
                        "Publishing the prediction and the residual makes each exoneration "
                        "falsifiable: if the prediction were wrong, the number on this page "
                        "would show it."
                    ),
                    "result": lines,
                }
            )

    conf = _confidence_detail(case)
    steps.append(
        {
            "id": "confidence",
            "title": f"Confidence — {case.confidence_value:.2f}",
            "state": "pass" if case.confidence_value >= 0.5 else "unknown",
            "score": f"{case.confidence_value:.2f}",
            "what": conf["what"],
            "why": (
                "A single number would hide which evidence is missing. Components that could "
                "not be measured withdraw their weight rather than scoring zero, and the total "
                "is capped in proportion to how much of the evidence was gathered, so a case "
                "measured on one test out of five cannot report high confidence."
            ),
            "result": conf["result"],
        }
    )

    return steps


def _confidence_detail(case: Case) -> dict[str, str]:
    try:
        data = json.loads(case.confidence_json or "{}")
    except (TypeError, ValueError):
        data = {}
    components = data.get("components") or []
    if not components:
        return {
            "what": "Scored the verdict against the evidence gathered.",
            "result": f"Confidence {case.confidence_value:.2f}. No component breakdown recorded.",
        }
    parts = [
        f"{c.get('name', '?')} {c.get('state', '?')}"
        + ("" if c.get("score") is None else f" at {float(c['score']):.2f}")
        + (f" (weight {float(c['weight']):.2f})" if c.get("weight") else "")
        for c in components
    ]
    caveat = data.get("caveat") or ""
    return {
        "what": (
            f"Combined {len(components)} components into a weighted score, renormalising over "
            "whichever ones could actually be measured."
        ),
        "result": "; ".join(parts) + (f". {caveat}" if caveat else ""),
    }


def _case_payload(case: Case) -> dict[str, Any]:
    finding = case.finding
    loc = case.localization
    accused = loc.accused
    impact = case.impact
    return {
        "id": case.case_id,
        "metric": finding.metric,
        "segment": (accused.segment if accused else finding.segment).label(),
        "verdict": case.verdict_kind,
        "detector": finding.detector,
        "confidence": round(case.confidence_value, 3),
        "effect": finding.relative_effect if accused is None else accused.relative_effect,
        "observed": _rate(finding.test.observed, finding),
        "expected": _rate(finding.test.expected, finding),
        "p": _sig(finding.test.p_value, 3),
        "revenue": None if impact.revenue is None else round(impact.revenue, 2),
        "revenue_direct": bool(getattr(impact, "direct", False)),
        "recurrence": case.recurrence_of or "",
        "narrative": case.narrative or "",
        "narrative_source": case.narrative_source,
        "steps": _steps_for(case),
        "candidates": [_candidate_payload(c, finding) for c in loc.candidates[:40]],
    }


def _candidate_payload(candidate: Candidate, finding: Finding) -> dict[str, Any]:
    return {
        "segment": candidate.segment.label(),
        "status": candidate.status,
        "effect": candidate.relative_effect,
        "observed": _rate(candidate.observed_value, finding),
        "expected": _rate(candidate.expected_value, finding),
        "reason": candidate.reason or "",
    }


def _gap_summary(gaps: list[CoverageGap]) -> list[dict[str, Any]]:
    """Coverage gaps rolled up by reason.

    Rolled up rather than listed: a run produces hundreds, and a wall of them would bury the
    one fact a reader needs, which is *why* cells went untested and how many. The individual
    rows are in the coverage_ledger table for anyone who wants them.
    """
    counts: dict[tuple[str, str], int] = {}
    for gap in gaps:
        key = (gap.metric, gap.reason)
        counts[key] = counts.get(key, 0) + 1
    rows = [
        {"metric": metric, "reason": reason, "count": n}
        for (metric, reason), n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return rows[:24]


def reproduction_sql(case: Case, table: str, expr: str) -> str:
    """A statement a reader can paste into ClickHouse to re-derive the headline numbers.

    The strongest form of traceability available: not a description of how the figure was
    computed but the computation itself, against the same rollup the detector read, returning
    the window beside the aligned weeks it was judged against. If this disagrees with the case
    file, the case file is wrong -- and a reader can find that out in one paste.
    """
    finding = case.finding
    segment = case.segment
    window = finding.window
    if segment.is_total:
        where = f"combo = '{_TOTAL}'"
    else:
        key_a, key_b = segment.combo_keys
        where = (
            f"combo = '{segment.combo}' AND key_a = '{_q(key_a)}' AND key_b = '{_q(key_b)}'"
        )
    start = f"toDateTime('{window.start:%Y-%m-%d %H:%M:%S}')"
    span = int((window.end - window.start).total_seconds())
    return (
        f"-- {finding.metric} for {segment.label()}\n"
        f"-- Row 0 is the window under investigation; rows 1..4 are the weekly-aligned windows\n"
        f"-- it was judged against. The detector compares row 0 to the median of the rest.\n"
        f"SELECT\n"
        # Ceiling division, not truncating. A bucket five days before the window start is part
        # of the window one week earlier, but truncating division files it under zero and it
        # lands in the group it is supposed to be compared against.
        f"    intDiv(dateDiff('second', bucket, {start}) + 604799, 604800) AS weeks_before,\n"
        # Aliases are prefixed because an alias matching a column name is resolved inside the
        # aggregate that references it, and ClickHouse rejects the result as nested aggregation.
        # The metric alias needs the same guard: every count metric is named after the very
        # column its own expression sums.
        f"    {expr} AS {_alias(finding.metric)},\n"
        f"    sum(requests) AS n_requests, sum(fills) AS n_fills,\n"
        f"    sum(impressions) AS n_impressions, sum(clicks) AS n_clicks, sum(revenue) AS n_revenue\n"
        f"FROM {table}\n"
        f"WHERE {where}\n"
        f"  AND bucket >= {start} - INTERVAL 4 WEEK\n"
        f"  AND bucket <  toDateTime('{window.end:%Y-%m-%d %H:%M:%S}')\n"
        # Keeps only the matching offset within each week, so a three-day window is compared
        # against the same three days a week earlier rather than against a whole week. Filtering
        # on toDayOfWeek instead would silently keep one day in three.
        f"  AND positiveModulo(dateDiff('second', {start}, bucket), 604800) < {span}\n"
        f"GROUP BY weeks_before\n"
        f"ORDER BY weeks_before"
    )


def _q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


#: Stored counter columns. Any alias equal to one of these is resolved inside the aggregate
#: that references it, which ClickHouse rejects outright.
_COUNTERS = ("requests", "fills", "impressions", "clicks", "revenue")


def _alias(name: str) -> str:
    return f"{name}_value" if name in _COUNTERS else name


_TOTAL = "__total__"

def _metric_expr(registry: Any, name: str) -> str:
    """The metric's rollup expression, or empty if it cannot be resolved.

    Empty rather than a guess. A case file with no provenance block is obviously incomplete;
    one showing a fabricated formula is confidently wrong, and a reader has no way to tell.
    """
    if registry is None:
        return ""
    try:
        return registry.metric(name).value_sql(from_rollup=True)
    except Exception:  # noqa: BLE001 - an unknown metric costs the panel, not the report
        return ""


def _spark(points: list[tuple[Any, float | None]]) -> list[dict[str, Any]]:
    return [
        {"t": t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t), "v": v}
        for t, v in points
    ]


def _metric_tree(result: InvestigationResult) -> list[dict[str, Any]]:
    """Every metric scanned, with the worst thing found in it.

    Amber rather than green for a metric that was scanned but whose cells were largely
    untestable. A metric nobody could measure is not a metric that came back clean, and showing
    both in the same colour is the single most misleading thing this page could do.
    """
    worst: dict[str, dict[str, Any]] = {
        name: {"metric": name, "state": "ok", "effect": 0.0, "segment": "", "cases": 0, "gaps": 0}
        for name in result.metrics_scanned
    }
    for gap in result.gaps:
        if gap.metric in worst:
            worst[gap.metric]["gaps"] += 1
    for case in result.cases:
        entry = worst.get(case.finding.metric)
        if entry is None:
            continue
        entry["cases"] += 1
        accused = case.localization.accused
        effect = accused.relative_effect if accused else case.finding.relative_effect
        if abs(effect) > abs(entry["effect"]):
            entry["effect"] = effect
            entry["segment"] = (accused.segment if accused else case.finding.segment).label()
        entry["state"] = "bad" if accused is not None else "warn"
    for entry in worst.values():
        if entry["cases"] == 0 and entry["gaps"] and not entry["segment"]:
            entry["state"] = "warn" if entry["gaps"] > 0 else "ok"
    return list(worst.values())


def build_payload(
    result: InvestigationResult,
    registry: Any = None,
    *,
    queries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Everything the page renders, as plain data.

    Split out from the HTML so the same structure can be asserted against in a test without
    parsing markup, and so a future consumer that is not a browser has something to read.

    The metric expression comes from the registry rather than being restated here. An earlier
    draft kept its own copy on the grounds that a reader wants readable SQL, which turned out
    to be a distinction without a difference -- the generated expression already reads as
    `sum(fills) / nullIf(sum(requests), 0)`. Keeping a second copy bought nothing and risked
    the one failure a reader could not detect: a query that runs, returns a plausible number,
    and quietly disagrees with the verdict printed beside it.
    """
    table = result.window.table
    cases = []
    for case in result.cases:
        payload = _case_payload(case)
        expr = _metric_expr(registry, case.finding.metric)
        payload["sql"] = reproduction_sql(case, table, expr) if expr else ""
        payload["series"] = _spark(getattr(case, "series", []) or [])
        cases.append(payload)
    return {
        "run_id": result.run_id,
        "window": result.window.label(),
        "grain": result.window.grain,
        "cells_tested": result.cells_tested,
        "findings": result.findings_after_correction,
        "metrics": result.metrics_scanned,
        "persisted": result.persisted,
        "tree": _metric_tree(result),
        "cases": cases,
        "gaps": _gap_summary(result.gaps),
        "gap_total": len(result.gaps),
        "queries": [{"name": k, "sql": v} for k, v in sorted((queries or {}).items())],
    }


def render(
    result: InvestigationResult, registry: Any = None, *, queries: dict[str, str] | None = None
) -> str:
    """The whole case file, as one string."""
    payload = build_payload(result, registry, queries=queries)
    data = json.dumps(payload).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data).replace("__TITLE__", _esc(payload["window"]))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verdict — __TITLE__</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --line:#262d38; --ink:#e6edf3; --dim:#8b949e;
  --ok:#3fb950; --bad:#f85149; --unknown:#d29922; --accent:#58a6ff; --chip:#1f2630;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1,h2,h3{margin:0;font-weight:600;letter-spacing:-.01em}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:var(--panel)}
header .sub{color:var(--dim);margin-top:4px}
.cards{display:flex;gap:12px;flex-wrap:wrap;padding:16px 24px;border-bottom:1px solid var(--line)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:132px}
.card .n{font-size:22px;font-weight:600}
.card .l{color:var(--dim);font-size:12px;margin-top:2px}
main{display:grid;grid-template-columns:300px 1fr 420px;height:calc(100vh - 190px)}
.col{overflow-y:auto;padding:16px}
.col+.col{border-left:1px solid var(--line)}
.caseitem{border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-bottom:9px;cursor:pointer;background:var(--panel)}
.caseitem:hover{border-color:var(--accent)}
.caseitem.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.caseitem .m{font-weight:600}
.caseitem .s{color:var(--dim);font-size:12.5px;word-break:break-word;margin-top:2px}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:6px}
.chip{background:var(--chip);border-radius:999px;padding:1px 9px;font-size:11.5px;color:var(--dim)}
.eff{font-weight:600}
.eff.dn{color:var(--bad)} .eff.up{color:var(--ok)}
.step{border:1px solid var(--line);border-left-width:3px;border-radius:9px;padding:10px 12px;margin-bottom:8px;cursor:pointer;background:var(--panel)}
.step:hover{border-color:var(--accent)}
.step.sel{box-shadow:0 0 0 1px var(--accent) inset}
.step.ok{border-left-color:var(--ok)} .step.bad{border-left-color:var(--bad)}
.step.unknown{border-left-color:var(--unknown)} .step.plain{border-left-color:var(--line)}
.step .t{font-weight:600;display:flex;justify-content:space-between;gap:10px}
.step .sc{color:var(--dim);font-weight:400}
.arrow{color:var(--line);text-align:center;margin:-4px 0 4px;font-size:15px}
.detail h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:16px 0 5px}
.detail h3:first-child{margin-top:0}
.detail p{margin:0}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:11.5px;font-weight:600}
.badge.ok{background:rgba(63,185,80,.15);color:var(--ok)}
.badge.bad{background:rgba(248,81,73,.15);color:var(--bad)}
.badge.unknown{background:rgba(210,153,34,.15);color:var(--unknown)}
.narr{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin-bottom:14px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:500}
.muted{color:var(--dim)}
.empty{color:var(--dim);padding:28px 12px;text-align:center}
.tree{display:flex;gap:8px;flex-wrap:wrap;padding:0 24px 16px}
.leaf{display:flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);
  border-left-width:3px;border-radius:8px;padding:7px 12px;font-size:12.5px}
.leaf.ok{border-left-color:var(--ok)} .leaf.warn{border-left-color:var(--unknown)}
.leaf.bad{border-left-color:var(--bad)}
.leaf .nm{font-weight:600}
.dot{width:7px;height:7px;border-radius:50%}
.dot.ok{background:var(--ok)} .dot.warn{background:var(--unknown)} .dot.bad{background:var(--bad)}
.sqlbox{background:#0b0e13;border:1px solid var(--line);border-radius:8px;padding:11px 13px;
  overflow-x:auto;white-space:pre;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px;line-height:1.5;color:#c9d1d9}
.copy{float:right;background:var(--chip);border:1px solid var(--line);color:var(--dim);
  border-radius:6px;padding:2px 9px;font-size:11px;cursor:pointer}
.copy:hover{color:var(--ink);border-color:var(--accent)}
svg.spark{display:block;width:100%;height:52px;margin:6px 0 2px}
details{margin-top:10px} summary{cursor:pointer;color:var(--dim);font-size:12.5px}
summary:hover{color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1>Verdict</h1>
  <div class="sub" id="sub"></div>
</header>
<div class="cards" id="cards"></div>
<div class="tree" id="tree"></div>
<main>
  <div class="col" id="cases"></div>
  <div class="col" id="flow"></div>
  <div class="col detail" id="detail"></div>
</main>
<script>
const DATA = __DATA__;
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[c]));
const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
const cls = s => ({pass:"ok", fail:"bad", unknown:"unknown"}[s] || "plain");
let sel = 0, selStep = 0;

document.getElementById("sub").textContent =
  DATA.window + " at " + DATA.grain + " grain · run " + DATA.run_id +
  (DATA.persisted ? "" : " · not persisted");

document.getElementById("cards").innerHTML = [
  [DATA.cases.length, "cases"],
  [DATA.cells_tested.toLocaleString(), "cells tested"],
  [DATA.findings, "survived correction"],
  [DATA.gap_total.toLocaleString(), "coverage gaps"],
  [DATA.metrics.length, "metrics scanned"],
].map(([n, l]) => `<div class="card"><div class="n">${esc(n)}</div><div class="l">${esc(l)}</div></div>`).join("");

// Metric tree. Amber is "scanned but largely untestable", which is not the same as clean and
// must not share a colour with it.
document.getElementById("tree").innerHTML = (DATA.tree || []).map(t => `
  <div class="leaf ${t.state}">
    <span class="dot ${t.state}"></span>
    <span class="nm">${esc(t.metric)}</span>
    ${t.segment ? `<span class="muted">${esc(t.segment)}</span>` : '<span class="muted">no case</span>'}
    ${t.effect ? `<span class="eff ${t.effect < 0 ? "dn" : "up"}">${pct(t.effect)}</span>` : ""}
    ${t.gaps ? `<span class="chip">${t.gaps} untestable</span>` : ""}
  </div>`).join("");

// Sparkline as inline SVG. No chart library: a self-contained file cannot fetch one, and a
// polyline is all a level-over-time needs.
function spark(points) {
  const vals = points.map(p => p.v).filter(v => v != null);
  if (vals.length < 2) return "";
  const lo = Math.min(...vals), hi = Math.max(...vals), span = (hi - lo) || 1;
  const w = 380, h = 52, pad = 3;
  const pts = points.map((p, i) => {
    const x = pad + (i / (points.length - 1)) * (w - 2 * pad);
    const y = p.v == null ? null : h - pad - ((p.v - lo) / span) * (h - 2 * pad);
    return y == null ? null : `${x.toFixed(1)},${y.toFixed(1)}`;
  }).filter(Boolean).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.6"/>
  </svg><div class="muted" style="font-size:11.5px">${esc(points[0].t)} &rarr; ${esc(points[points.length-1].t)} · low ${lo.toFixed(4)} · high ${hi.toFixed(4)}</div>`;
}

function copySql(i) {
  navigator.clipboard && navigator.clipboard.writeText(DATA.cases[i].sql);
}

function renderCases() {
  const el = document.getElementById("cases");
  if (!DATA.cases.length) { el.innerHTML = '<div class="empty">No incident met the reporting bar.</div>'; return; }
  el.innerHTML = DATA.cases.map((c, i) => `
    <div class="caseitem ${i === sel ? "sel" : ""}" onclick="pick(${i})">
      <div class="m">${esc(c.metric)}</div>
      <div class="s">${esc(c.segment)}</div>
      <div class="row">
        <span class="eff ${c.effect < 0 ? "dn" : "up"}">${pct(c.effect)}</span>
        <span class="chip">conf ${c.confidence.toFixed(2)}</span>
      </div>
      <div class="row">
        <span class="chip">${esc(c.verdict)}</span>
        ${c.revenue != null ? `<span class="chip">${c.revenue_direct ? "" : "~"}${Number(c.revenue).toLocaleString()}</span>` : ""}
      </div>
    </div>`).join("");
}

function renderFlow() {
  const c = DATA.cases[sel];
  const el = document.getElementById("flow");
  if (!c) { el.innerHTML = ""; return; }
  const steps = c.steps.map((s, i) => `
    <div class="step ${cls(s.state)} ${i === selStep ? "sel" : ""}" onclick="pickStep(${i})">
      <div class="t"><span>${esc(s.title)}</span><span class="sc">${esc(s.score)}</span></div>
    </div>${i < c.steps.length - 1 ? '<div class="arrow">&darr;</div>' : ""}`).join("");
  const cands = c.candidates.length ? `
    <h3 class="muted" style="margin-top:18px">Candidates considered</h3>
    <table><tr><th>Segment</th><th>Status</th><th>Effect</th></tr>
    ${c.candidates.map(x => `<tr><td>${esc(x.segment)}</td><td>${esc(x.status)}</td><td>${pct(x.effect)}</td></tr>`).join("")}
    </table>` : "";
  el.innerHTML = steps + cands;
}

function renderDetail() {
  const c = DATA.cases[sel];
  const el = document.getElementById("detail");
  if (!c) { el.innerHTML = ""; return; }
  const s = c.steps[selStep];
  const narr = c.narrative
    ? `<div class="narr">${esc(c.narrative)}</div><div class="muted" style="margin-bottom:16px">Narration source: ${esc(c.narrative_source)}</div>`
    : "";
  const chart = c.series && c.series.length
    ? `<h3>${esc(c.metric)} over time — ${esc(c.segment)}</h3>${spark(c.series)}` : "";
  const step = s ? `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
      <h2>${esc(s.title)}</h2>
      <span class="badge ${cls(s.state)}">${esc(s.state)}</span>
    </div>
    <h3>What was done</h3><p>${esc(s.what)}</p>
    <h3>Why</h3><p>${esc(s.why)}</p>
    <h3>What it gave us</h3><p>${esc(s.result)}</p>` : "";
  // Provenance last, because it is what a reader reaches for after they have decided they do
  // not believe something.
  const sql = c.sql ? `
    <h3>Check it yourself
      <button class="copy" onclick="copySql(${sel})">copy</button>
    </h3>
    <div class="sqlbox">${esc(c.sql)}</div>
    <div class="muted" style="margin-top:6px;font-size:12px">
      Runs against the same rollup the detector read. If it disagrees with this page, this page is wrong.
    </div>` : "";
  const shapes = (DATA.queries && DATA.queries.length) ? `
    <details><summary>Every query shape this run executed (${DATA.queries.length})</summary>
      ${DATA.queries.map(q => `<h3>${esc(q.name)}</h3><div class="sqlbox">${esc(q.sql)}</div>`).join("")}
    </details>` : "";
  el.innerHTML = narr + chart + step + sql + shapes;
}

function pick(i) { sel = i; selStep = 0; draw(); }
function pickStep(i) { selStep = i; draw(); }
function draw() { renderCases(); renderFlow(); renderDetail(); }
draw();
</script>
</body>
</html>"""


__all__ = ["build_payload", "render"]
