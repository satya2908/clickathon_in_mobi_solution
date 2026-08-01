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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run one incident by id, e.g. F")
    parser.add_argument("--grain", default="1h")
    args = parser.parse_args()

    cfg = load_config()
    registry = MetricRegistry.load(os.environ.get("VERDICT_METRICS") or "config/metrics.yaml")
    ch = ClickHouse(cfg.clickhouse)

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
