"""Checking that the baseline still describes the population before trusting it.

Every temporal verdict rests on one unstated assumption: that a cell's own recent history
predicts what it should be doing now. That assumption is usually safe and occasionally false,
and when it is false nothing downstream notices. The significance tests still run, the
counterfactuals still pass, the confidence components still score -- against an expectation
drawn from a population that no longer exists. The output is not noisy. It is confident,
internally consistent, and wrong.

This is not hypothetical. A dataset can reissue its dimension tables with the same identifiers
and different attribute values, at which point `publisher_tier=tier_3` before the boundary and
`publisher_tier=tier_3` after it name two different groups of apps. History is intact, the
label is intact, and the comparison between them is meaningless.

The check is calibration rather than prediction accuracy. Ask the detector to scan a recent
window and count what share of the grid it flags. A sound baseline flags roughly the
false-discovery rate plus whatever genuinely happened -- low single digits. A baseline
describing the wrong population flags a large fraction of everything, because almost every cell
really does differ from an expectation built for someone else. Those two regimes are orders of
magnitude apart, so the reading does not need to be delicate to be decisive.

What makes this usable is that it needs no labels. It never has to know which day was normal.

When the audit fails, the temporal detector is switched off for the run rather than
second-guessed. The structural detector keeps working, because it compares each cell against
its siblings inside the same bucket and consults no history at all -- so it is untouched by
whatever made the history incomparable, and it is the detector that can still see a segment
sitting at half the fill rate of everything beside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config
from .detect import DetectionResult, apply_correction, detect_temporal, lattice_combos
from .metrics import MetricRegistry
from .query import RollupReader, Window

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineAudit:
    """What the grid says about whether segment-level history can be believed."""

    trustworthy: bool
    flagged_rate: float
    bar: float
    windows: tuple[str, ...] = ()
    rates: tuple[float, ...] = ()
    tested: int = 0
    ran: bool = True
    #: Share of the grid the baseline disagrees with over the window being investigated. Read
    #: separately from the historical windows because the two answer different questions.
    target_rate: float = 0.0
    target_tested: int = 0
    #: Which reading rejected: "history", "boundary", or "" when nothing did.
    reason: str = ""

    @property
    def headline(self) -> str:
        if not self.ran:
            return "Baseline not audited."
        bar = f"{self.bar:.0%}"
        if self.trustworthy:
            return (
                f"Baseline calibrated: {self.flagged_rate:.1%} of cells flagged on a recent "
                f"window, under {bar}."
            )
        if self.reason == "boundary":
            return (
                f"Baseline rejected at this window: {self.target_rate:.1%} of the grid disagrees "
                f"with history here against a bar of {bar}, while recent windows sat at "
                f"{self.flagged_rate:.1%}."
            )
        return (
            f"Baseline rejected: {self.flagged_rate:.1%} of all tested cells are flagged on a "
            f"recent window, against a bar of {bar}."
        )

    @property
    def detail(self) -> str:
        if not self.ran or self.trustworthy:
            return self.headline
        if self.reason == "boundary":
            cause = (
                "History was sound until this window and stops describing the population inside "
                "it. That is the shape of a corpus boundary -- dimension attributes reissued "
                "against the same identifiers, so a label names one group of entities before it "
                "and a different group after -- rather than of an incident, which moves a small "
                "part of the grid rather than most of it."
            )
        else:
            cause = (
                "A baseline that disagrees with that much of the grid is not describing this "
                "population, so every segment-level comparison drawn from it -- the expected "
                "values, the effect sizes and the significance -- would be measured against the "
                "wrong thing."
            )
        return (
            f"{self.headline} {cause} Segment-level temporal detection is therefore switched off "
            f"for this run. The platform aggregate is still compared against its own history, "
            f"because relabelling which entities carry which attribute cannot move a total, and "
            f"localization falls to the structural comparison against siblings in the same "
            f"window, which consults no history at all."
        )


def flagged_share(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: Config,
    window: Window,
    names: list[str],
) -> tuple[int, int]:
    """Fraction of the temporal grid that survives correction over one window.

    Returns (survivors, tested). This is the same scan `detect_all` performs, restricted to the
    temporal detector, because the structural one has no baseline to audit.
    """
    wanted: list[str] = []
    for name in names:
        wanted.extend(lattice_combos(registry, registry.metric(name), window.grain))
    reader.prefetch_lattice(wanted, window, cfg.detection.baseline_weeks)

    scan = DetectionResult()
    for name in names:
        try:
            found = detect_temporal(
                reader, registry, cfg.detection, name, window, correct=False
            )
        except Exception as exc:  # noqa: BLE001 - an audit must not be able to end a run
            log.warning("Baseline audit could not scan %s over %s: %s", name, window.label(), exc)
            continue
        # extend(), so tested_cells reaches the correction and the family is sized correctly.
        scan.extend(found)

    tested = len(scan.findings)
    if not tested:
        return 0, 0
    return len(apply_correction(scan, cfg.detection).findings), tested


def audit_baseline(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: Config,
    window: Window,
    *,
    metrics: list[str] | None = None,
) -> BaselineAudit:
    """Decide whether segment-level history can be trusted for this window.

    Two readings, because they fail differently.

    The first audits the windows immediately before the one under test and takes the *best* of
    them. Taking the minimum rather than the mean is deliberate: a genuine incident inflates the
    flagged share of the window containing it, while a broken baseline inflates every window at
    once, so requiring all of them to look wrong keeps an incident from being mistaken for
    miscalibration.

    That reading alone cannot see a boundary until the boundary is behind it. A corpus that
    reissues its dimensions between one day and the next leaves every prior window clean and
    every comparison across the seam meaningless, so the check stays silent for exactly as many
    days as it is wide -- and those are the days whose verdicts are worth least. The second
    reading closes that by measuring the window under test directly.

    A high reading there is ambiguous on its own: it is either a boundary or an incident large
    enough to move most of the grid. What makes it safe to act on anyway is that acting is now
    cheap. Rejecting no longer means going blind -- the platform aggregate keeps its own history,
    which no relabelling can disturb, and localization falls to sibling comparison, which never
    had a baseline. Both survivors work on an incident too. So the ambiguity costs nothing, and
    the reading is decisive rather than conservative.

    For scale: on this corpus a clean day disagrees with 0.2% to 2.2% of the grid and the day a
    reissued dimension set arrives disagrees with 42%. The bar does not need to be delicate.
    """
    names = metrics or list(registry.metrics)
    bar = cfg.detection.baseline_audit_max_flagged

    rates: list[float] = []
    labels: list[str] = []
    total_tested = 0

    for step in range(1, max(1, cfg.detection.baseline_audit_windows) + 1):
        shift = window.duration * step
        prior = Window(start=window.start - shift, end=window.end - shift, grain=window.grain)
        survivors, tested = flagged_share(reader, registry, cfg, prior, names)
        if not tested:
            continue
        rates.append(survivors / tested)
        labels.append(prior.label())
        total_tested += tested
        log.info(
            "Baseline audit on %s: %s of %s cells flagged (%.1f%%)",
            prior.label(), f"{survivors:,}", f"{tested:,}", 100 * survivors / tested,
        )

    target_survivors, target_tested = flagged_share(reader, registry, cfg, window, names)
    target_rate = target_survivors / target_tested if target_tested else 0.0
    if target_tested:
        log.info(
            "Baseline audit on the window under test (%s): %s of %s cells flagged (%.1f%%)",
            window.label(), f"{target_survivors:,}", f"{target_tested:,}", 100 * target_rate,
        )

    if not rates:
        # Nothing to audit against -- the corpus does not reach back far enough. Say so rather
        # than reading an absent check as a passed one.
        return BaselineAudit(
            trustworthy=True, flagged_rate=0.0, bar=bar, ran=False,
            target_rate=target_rate, target_tested=target_tested,
        )

    best = min(rates)
    # History first, so a baseline that has been wrong for a while is named as that rather than
    # as a boundary that happens to sit at this window.
    reason = "history" if best > bar else "boundary" if target_rate > bar else ""
    return BaselineAudit(
        trustworthy=not reason,
        flagged_rate=best,
        bar=bar,
        windows=tuple(labels),
        rates=tuple(rates),
        tested=total_tested,
        target_rate=target_rate,
        target_tested=target_tested,
        reason=reason,
    )
