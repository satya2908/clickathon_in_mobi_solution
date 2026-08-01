"""Score the analyst against the movements known to be in the development corpus.

This is the number that matters, and it is deliberately kept separate from the unit tests. The
tests check that each component does what its author intended; this checks whether the assembled
system reaches the right conclusion on data whose answer is known independently of the code.

The answer key below was established by direct measurement of the corpus, not by running this
system, which is the only way a score from it means anything. Two of the six entries are here
specifically because they are expected to be hard, and a run that quietly stopped reporting them
should show up as a regression rather than as an improvement in the average.

Usage:
    LLM_ENABLED=false python scripts/accuracy.py
    LLM_ENABLED=false python scripts/accuracy.py --only F
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from verdict.config import load_config  # noqa: E402
from verdict.db import ClickHouse  # noqa: E402
from verdict.metrics import MetricRegistry  # noqa: E402
from verdict.pipeline import investigate  # noqa: E402
from verdict.query import Window  # noqa: E402
from verdict.trace import NullTracer  # noqa: E402


@dataclass(frozen=True)
class Known:
    ident: str
    metric: str
    start: datetime
    hours: int
    expect: dict[str, str]
    change: float
    shape: str
    # A movement the design does not claim to localize. Recorded so a miss reads as a known
    # limitation rather than as an unexplained failure, and so that finding one is visible as a
    # genuine gain instead of disappearing into the average.
    stretch: bool = False
    also: tuple[dict[str, str], ...] = field(default_factory=tuple)


ANSWER_KEY = [
    Known("A", "fill_rate", datetime(2026, 6, 23), 48, {"os_version": "Android 15"}, -0.448, "main effect"),
    Known(
        "B", "fill_rate", datetime(2026, 6, 28), 48,
        {"region": "APAC", "os_version": "iOS 18.1"}, -0.508, "interaction",
    ),
    Known("C", "ecpm", datetime(2026, 6, 19), 72, {"category": "finance"}, -0.350, "main effect"),
    Known("D", "requests", datetime(2026, 6, 21), 24, {}, -0.435, "uniform, global"),
    Known(
        "E", "ctr", datetime(2026, 6, 16), 240, {"publisher_tier": "tier_3"}, -0.220,
        "two-phase, small", stretch=True,
    ),
    Known(
        "F", "ecpm", datetime(2026, 6, 16), 72,
        {"region": "EU", "ad_format": "interstitial"}, -0.300, "compensating pair",
        also=({"region": "EU", "ad_format": "native"},),
    ),
]


RANK = {"exact": 5, "too_narrow": 4, "too_broad": 3, "detected": 2, "wrong": 1, "none": 0}


def outcome_for(case: object, expected: dict[str, str]) -> str:
    """How a finished case relates to the truth.

    The distinction that has to be preserved here is between a case the detector raised and a
    case the localizer actually attributed. Reading the segment off an unattributed case reports
    the cell the detector happened to enter on, which flatters the score badly: it scores the
    detector's work as though the localizer had done it, and it hides a total failure to
    attribute behind a correct-looking segment name.

    For a genuinely global movement the correct behaviour is to accuse nobody, so declining is
    scored as exact and naming a culprit is scored as a false accusation rather than as a near
    miss. Sending an operator to inspect one category during a platform-wide outage is worse
    than saying nothing.
    """
    accused = case.localization.accused
    if not expected:
        return "exact" if accused is None else "false_culprit"
    if accused is None:
        return "detected"

    named = accused.segment.as_dict()
    if named == expected:
        return "exact"
    if all(named.get(k) == v for k, v in expected.items()):
        return "too_narrow"
    if all(expected.get(k) == v for k, v in named.items()):
        return "too_broad"
    return "wrong"


# Windows carrying no movement anyone planted. Recall says how often the system finds what is
# there; only a control says how often it announces something that is not, and a system with an
# unmeasured false-positive rate has an unmeasured precision no matter how good its recall looks.
#
# These are derived by subtraction rather than by inspection: the corpus runs 1 June to 5 July and
# every known movement falls between 16 June and 30 June, so what remains at either end is clean
# without anyone having examined it for quietness. Picking windows because they looked calm would
# make this measure nothing at all.
# The split matters as much as the windows. A weekly baseline needs at least two aligned
# samples before the spread between them can be measured at all, and with the corpus opening on
# 1 June that arrives on 15 June. Windows before it are kept deliberately: the system should now
# decline to test them and say so, and a run where they start producing verdicts again is a
# regression that the headline number alone would hide.
CONTROL_WINDOWS = [
    ("thin-1", datetime(2026, 6, 8), 24, False),
    ("thin-2", datetime(2026, 6, 10), 24, False),
    ("thin-3", datetime(2026, 6, 12), 24, False),
    ("full-1", datetime(2026, 6, 15), 24, True),
    ("full-2", datetime(2026, 7, 1), 24, True),
    ("full-3", datetime(2026, 7, 2), 24, True),
    ("full-4", datetime(2026, 7, 3), 24, True),
    ("full-5", datetime(2026, 7, 4), 24, True),
]


def run_controls(cfg, registry, ch, grain: str, threshold: float) -> int:
    """Count confident accusations in windows where nothing was planted.

    Every accusation here is a false positive by construction. Unattributed cases are counted
    separately: raising a detection the localizer then declines to attribute is a far cheaper
    error than naming an innocent segment, and collapsing the two would hide which one the
    system is actually making.
    """
    print("\n" + "=" * 108)
    print("CONTROLS -- windows with nothing planted in them. Every accusation below is a false positive.")
    print("-" * 108)
    print(f"{'window':<10} {'date':<12} {'accused':<11} {'unattributed':<13} {'worst false accusation':<45}")
    print("-" * 108)

    tally = {True: [0, 0, 0], False: [0, 0, 0]}  # accused, unattributed, windows
    leaked = []
    for name, start, hours, testable in CONTROL_WINDOWS:
        window = Window(start=start, end=start + timedelta(hours=hours), grain=grain)
        result = investigate(
            cfg, ch, registry, window, tracer=NullTracer(), persist=False, narrate=False
        )
        accused = [c for c in result.cases if c.localization.accused is not None]
        confident = [c for c in accused if c.confidence_value >= threshold]
        unattributed = len(result.cases) - len(accused)

        worst = ""
        if confident:
            top = max(confident, key=lambda c: c.confidence_value)
            worst = f"{top.finding.metric} {top.segment.label()[:26]} conf={top.confidence_value:.2f}"

        bucket = tally[testable]
        bucket[0] += len(confident)
        bucket[1] += unattributed
        bucket[2] += 1
        # Only the temporal detector needs a dispersion estimate, and it is the only one the
        # guard can silence. The structural detector compares a cell against its siblings inside
        # the same window and holds no opinion about history, so it is *expected* to speak in a
        # window with one baseline week. Counting its output as a leak said the guard had failed
        # when the guard had in fact silenced all 15,236 testable cells it was asked about.
        if not testable:
            leaked += [c for c in result.cases if c.finding.detector == "temporal"]
        print(f"{name:<10} {start:%Y-%m-%d}   {len(confident):<11} {unattributed:<13} {worst:<45}")

    print("-" * 108)
    for testable, label in ((True, "enough baseline to test"), (False, "too little baseline")):
        acc, unatt, n = tally[testable]
        if not n:
            continue
        print(
            f"{label:<26} {n} day(s): {acc} false accusation(s) at conf >= {threshold:.2f} "
            f"({acc / n:.2f}/day), {unatt} unattributed ({unatt / n:.2f}/day)"
        )
    if tally[False][2]:
        print(
            "\nThin windows: output there is structural only -- sibling comparison inside the "
            "window, which needs no history and is expected to speak."
        )
    if leaked:
        print(
            f"\nWARNING: {len(leaked)} temporal case(s) in windows with under two aligned "
            "baseline weeks. The dispersion estimate cannot be formed there, so the temporal "
            "detector should have been silent: " + ", ".join(sorted({c.finding.metric for c in leaked}))
        )
    print("=" * 108 + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run one incident by id, e.g. F")
    parser.add_argument("--grain", default="1h")
    parser.add_argument("--controls", action="store_true", help="Measure the false-positive rate instead")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence at which an accusation counts")
    args = parser.parse_args()

    cfg = load_config()
    registry = MetricRegistry.load(os.environ.get("VERDICT_METRICS") or "config/metrics.yaml")
    ch = ClickHouse(cfg.clickhouse)

    if args.controls:
        return run_controls(cfg, registry, ch, args.grain, args.threshold)

    key = [k for k in ANSWER_KEY if not args.only or k.ident == args.only.upper()]
    rows = []

    for known in key:
        window = Window(
            start=known.start, end=known.start + timedelta(hours=known.hours), grain=args.grain
        )
        result = investigate(
            cfg, ch, registry, window,
            metrics=[known.metric], tracer=NullTracer(), persist=False, narrate=False,
        )

        best = ("none", None, 0.0, 0.0)
        for case in result.cases:
            outcome = outcome_for(case, known.expect)
            if RANK.get(outcome, 1) <= RANK.get(best[0], 0):
                continue
            accused = case.localization.accused
            effect = accused.relative_effect if accused else case.finding.test.relative_effect
            label = accused.segment.label() if accused else f"({case.segment.label()})"
            best = (outcome, label, case.confidence_value, effect)

        found_second = ""
        for extra in known.also:
            for case in result.cases:
                accused = case.localization.accused
                if accused is not None and accused.segment.as_dict() == extra:
                    found_second = accused.segment.label()
        rows.append((known, best, found_second, len(result.cases), len(result.gaps)))

    width = 108
    print("\n" + "=" * width)
    print(f"{'ID':<3} {'metric':<10} {'shape':<18} {'outcome':<11} {'accused':<30} {'conf':>5} {'effect':>8}")
    print("-" * width)
    exact = 0
    for known, (outcome, accused, conf, effect) in [(r[0], r[1]) for r in rows]:
        flag = " (stretch)" if known.stretch else ""
        if outcome == "exact":
            exact += 1
        print(
            f"{known.ident:<3} {known.metric:<10} {known.shape[:17]:<18} {outcome:<11} "
            f"{(accused or '-')[:29]:<30} {conf:>5.2f} {effect:>+8.1%}{flag}"
        )
    print("-" * width)
    print(f"exact localization: {exact}/{len(rows)}")
    print("a name in (parentheses) is the detector's cell on a case the localizer did not attribute")
    print("truth for reference:")
    for known, _, second, cases, gaps in rows:
        extra = f"  second leg found: {second}" if second else ("  second leg NOT found" if known.also else "")
        print(f"  {known.ident}: truth {known.change:+.1%} in {known.expect or 'global'}"
              f"   [{cases} case(s), {gaps} gap(s)]{extra}")
    print("=" * width + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
