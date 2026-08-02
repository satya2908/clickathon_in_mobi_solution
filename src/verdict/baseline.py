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
    """What a recent window says about whether the baseline can be believed."""

    trustworthy: bool
    flagged_rate: float
    bar: float
    windows: tuple[str, ...] = ()
    rates: tuple[float, ...] = ()
    tested: int = 0
    ran: bool = True

    @property
    def headline(self) -> str:
        if not self.ran:
            return "Baseline not audited."
        pct, bar = f"{self.flagged_rate:.1%}", f"{self.bar:.0%}"
        if self.trustworthy:
            return f"Baseline calibrated: {pct} of cells flagged on a recent window, under {bar}."
        return (
            f"Baseline rejected: {pct} of all tested cells are flagged on a recent window, "
            f"against a bar of {bar}."
        )

    @property
    def detail(self) -> str:
        if not self.ran or self.trustworthy:
            return self.headline
        return (
            f"{self.headline} A baseline that disagrees with that much of the grid is not "
            f"describing this population, so every temporal comparison drawn from it -- the "
            f"expected values, the effect sizes and the significance -- would be measured "
            f"against the wrong thing. Temporal detection is therefore switched off for this "
            f"run and only the structural detector, which uses no history, is reported."
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
    """Decide whether the temporal baseline can be trusted for this window.

    Audits the windows immediately before the one under test, then takes the *best* of them.
    Taking the minimum rather than the mean is deliberate and is the difference between a
    useful check and one that switches the detector off during outages: a genuine incident
    inflates the flagged share of the window containing it, while a broken baseline inflates
    every window at once. Requiring every recent window to look wrong before declaring the
    baseline wrong keeps real incidents from being mistaken for miscalibration.
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

    if not rates:
        # Nothing to audit against -- the corpus does not reach back far enough. Say so rather
        # than reading an absent check as a passed one.
        return BaselineAudit(
            trustworthy=True, flagged_rate=0.0, bar=bar, ran=False,
        )

    best = min(rates)
    return BaselineAudit(
        trustworthy=best <= bar,
        flagged_rate=best,
        bar=bar,
        windows=tuple(labels),
        rates=tuple(rates),
        tested=total_tested,
    )
