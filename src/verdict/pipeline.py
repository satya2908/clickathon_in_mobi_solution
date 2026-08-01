"""The end-to-end investigation: detect, correct, localize, score, narrate, persist.

This module is deliberately thin. Every decision that affects a verdict was made in the modules
it calls, and the value of keeping the orchestration separate is that the order of operations
becomes something you can read in one screen and argue with.

Two orderings in here are not interchangeable and are worth stating plainly.

Correction happens after every detector has run and before anything is localized. Localizing
first would waste the expensive step on cells that the false-discovery-rate control is about to
throw away, and correcting per metric would leave the overall error rate multiplied by the
number of metrics.

Localization happens once per incident, not once per finding. A single fill-rate collapse in one
country produces a finding at the total, another at the country, and another at every pair
containing it, because the detector scans the whole lattice and all of those cells really did
move. They are one incident. Emitting three cases for them would be a reporting bug that looks
like three problems.

A known limitation, stated here rather than buried: the Benjamini-Hochberg family is the temporal
detector's, because that detector returns every cell it tested and BH needs the complete family
to be meaningful. The structural detector applies a fixed z threshold and an effect floor before
returning, so its survivors are not FDR-controlled and are marked as such on the case rather than
being quietly pooled into a family they do not belong to.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import Config
from .db import ClickHouse
from .detect import CoverageGap, DetectionResult, Finding, apply_correction, detect_temporal
from .localize import Localization, Localizer
from .metrics import MetricRegistry
from .query import RollupReader, Window
from .store import Case, CaseStore, build_case, direction_of
from .structural import detect_structural
from .trace import NullTracer, Tracer

log = logging.getLogger(__name__)


def _optional_confidence() -> Any | None:
    try:
        from . import confidence as module
    except ImportError:
        return None
    return module


def _optional_narrate() -> Any | None:
    try:
        from . import narrate as module
    except ImportError:
        return None
    return module


def git_sha() -> str:
    """The commit the run was produced by, so a case can be reproduced from source.

    Best-effort by design. A run from an exported tarball with no git metadata is still a valid
    run, and refusing to investigate because provenance is unavailable would be absurd.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class InvestigationResult:
    run_id: str
    window: Window
    cases: list[Case] = field(default_factory=list)
    gaps: list[CoverageGap] = field(default_factory=list)
    cells_tested: int = 0
    findings_after_correction: int = 0
    metrics_scanned: list[str] = field(default_factory=list)
    persisted: bool = False

    @property
    def publishable(self) -> list[Case]:
        return [c for c in self.cases if c.confidence_value >= 0.0 and c.verdict_kind == "localized"]

    def summary(self) -> str:
        return (
            f"run {self.run_id}: {self.cells_tested:,} cells tested across "
            f"{len(self.metrics_scanned)} metrics, {self.findings_after_correction} findings "
            f"survived correction, {len(self.cases)} case(s), {len(self.gaps)} coverage gap(s)"
        )


def verdict_key(case: Case) -> tuple[str, str, str, str]:
    """What makes two finished cases the same conclusion.

    Grouping findings before localization is not enough on its own. The localizer re-derives its
    candidates from the total every time, so two groups entered from different findings can
    arrive at the same accused segment and would otherwise be published as two incidents that
    happen to look identical. Deduplicating on the conclusion rather than on the entry point
    catches that, and it does so without assuming the two entry findings were related.
    """
    accused = case.localization.accused
    effect = accused.relative_effect if accused else case.finding.test.relative_effect
    return (case.finding.metric, case.finding.window.label(), case.segment.label(), direction_of(effect))


def _finding_for_accused(
    localization: Localization, group: list[Finding], everything: list[Finding]
) -> Finding | None:
    """The finding that describes the segment actually accused, if one was tested.

    A case carries a finding for its statistics -- the detector that raised it, the p-value, the
    overdispersion, how many baseline weeks survived trimming -- and those numbers reach the case
    file and the narration. The finding a group is entered on is whichever cell had the smallest
    p-value, which is routinely a two-dimensional cell *inside* the segment that localization
    eventually names. Quoting it produces a case that accuses one segment while reporting another
    segment's test: on this corpus a fill-rate verdict on Android 15 cited a structural anomaly in
    India on Galaxy S23, and reported a baseline of zero weeks because structural findings carry
    no weekly baseline at all.

    Localization is what decides the answer, so once it has, the case should quote the accused's
    own test. Preferring a temporal finding matters because only that detector measures a segment
    against its own history, which is what the narration describes.
    """
    accused = localization.accused
    if accused is None:
        return None
    for pool in (group, everything):
        matches = [f for f in pool if f.segment == accused.segment]
        if matches:
            matches.sort(key=lambda f: (f.detector != "temporal", f.test.p_value))
            return matches[0]
    return None


def _survivor_rank(case: Case) -> tuple[int, float, float]:
    """How good a representative a case is of the conclusion it reached.

    Coherence comes before confidence, and that ordering is the point. A case carries both the
    finding it was entered on and the segment it ended up accusing, and those need not be the
    same cell: several findings collapse to one conclusion, and whichever survives supplies the
    statistics the case file will quote. Choosing purely on confidence can therefore leave a case
    accusing one segment while quoting the test statistics of another, which is how a fill-rate
    verdict about Android 15 ends up citing a structural anomaly in a different cell and
    reporting a baseline of zero weeks.

    So the entry finding that describes the accused segment itself wins, and only then does
    strength of evidence break the tie.
    """
    accused = case.localization.accused
    coherent = 1 if accused is not None and case.finding.segment == accused.segment else 0
    return (coherent, case.confidence_value, -case.finding.test.p_value)


def dedupe_cases(cases: list[Case]) -> list[Case]:
    """Keep one case per distinct conclusion, preferring the best representative of it."""
    best: dict[tuple[str, str, str, str], Case] = {}
    for case in cases:
        key = verdict_key(case)
        incumbent = best.get(key)
        if incumbent is None or _survivor_rank(case) > _survivor_rank(incumbent):
            best[key] = case
    return list(best.values())


def _incident_key(finding: Finding) -> tuple[str, str, str]:
    """What makes two findings the same incident.

    Direction is part of the key so that a compensating pair -- one segment falling while another
    rises by the amount that hides it in the total -- is reported as the two problems it is
    rather than averaged into one confusing case.
    """
    return (finding.metric, finding.window.label(), direction_of(finding.test.relative_effect))


def group_findings(findings: list[Finding]) -> list[list[Finding]]:
    """Collapse the lattice's many views of one incident into one group each.

    Ordered by the strongest evidence in each group so that the most significant incident is
    investigated first and a truncated run still reports the things that matter most.
    """
    buckets: dict[tuple[str, str, str], list[Finding]] = {}
    for finding in findings:
        buckets.setdefault(_incident_key(finding), []).append(finding)

    groups = []
    for group in buckets.values():
        group.sort(key=lambda f: (f.test.p_value, -abs(f.test.relative_effect)))
        groups.append(group)
    groups.sort(key=lambda g: (g[0].test.p_value, -abs(g[0].test.relative_effect)))
    return groups


def detect_all(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: Config,
    window: Window,
    *,
    metrics: list[str] | None = None,
    tracer: Tracer | None = None,
    structural: bool = True,
) -> tuple[DetectionResult, DetectionResult]:
    """Run both detectors over every requested metric, returning them unmixed.

    They are kept apart because only one of them can be FDR-corrected honestly; see the module
    docstring. Callers that want a single list should correct the temporal result first and then
    concatenate, which is what ``investigate`` does.
    """
    names = metrics or list(registry.metrics)
    temporal = DetectionResult()
    struct = DetectionResult()

    for name in names:
        result = detect_temporal(reader, registry, cfg.detection, name, window, tracer=tracer, correct=False)
        temporal.findings.extend(result.findings)
        temporal.gaps.extend(result.gaps)

        if not structural:
            continue
        try:
            found = detect_structural(reader, registry, cfg.detection, name, window, tracer=tracer)
        except Exception as exc:  # noqa: BLE001 - one metric's failure must not end the scan
            log.warning("Structural scan failed for %s: %s", name, exc)
            continue
        struct.findings.extend(found.findings)
        struct.gaps.extend(found.gaps)

    return temporal, struct


def investigate(
    cfg: Config,
    ch: ClickHouse,
    registry: MetricRegistry,
    window: Window,
    *,
    metrics: list[str] | None = None,
    tracer: Tracer | None = None,
    persist: bool = True,
    narrate: bool = True,
    max_cases: int = 25,
) -> InvestigationResult:
    """Investigate one window and return everything concluded about it."""
    tracer = tracer or NullTracer()
    run_id = tracer.run_id or uuid.uuid4().hex
    started = datetime.now(UTC)
    reader = RollupReader(ch)
    store = CaseStore(ch)
    result = InvestigationResult(run_id=run_id, window=window)
    result.metrics_scanned = metrics or list(registry.metrics)

    if persist:
        try:
            store.open_run(run_id, config_json=cfg.redacted_json(), git_sha=git_sha())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open run row: %s", exc)

    with tracer.span("detect", kind="detector") as span:
        span.what(f"Scanned {len(result.metrics_scanned)} metric(s) over {window.label()} at {window.grain}")
        span.why(
            "Every cell in the lattice is compared against its own history and against its "
            "siblings, because an incident confined to one cell is invisible in the total."
        )
        temporal, struct = detect_all(
            reader, registry, cfg, window, metrics=metrics, tracer=tracer, structural=True
        )
        result.cells_tested = len(temporal.findings)
        span.result(
            f"{len(temporal.findings):,} cells tested, {len(struct.findings)} structural "
            f"anomalies, {len(temporal.gaps) + len(struct.gaps)} could not be tested"
        )

    with tracer.span("correct", kind="statistics") as span:
        span.what(
            f"Benjamini-Hochberg at alpha={cfg.detection.p_value_threshold} "
            f"over {len(temporal.findings):,} tests"
        )
        span.why(
            "At an uncorrected threshold, one cell in a hundred crosses it by chance, which "
            "across this lattice means dozens of confident findings in data where nothing "
            "happened."
        )
        temporal = apply_correction(temporal, cfg.detection)
        span.result(f"{len(temporal.findings)} finding(s) survived")

    findings = list(temporal.findings) + list(struct.findings)
    result.findings_after_correction = len(findings)
    result.gaps = list(temporal.gaps) + list(struct.gaps)

    localizer = Localizer(reader, registry, cfg.localization, cfg.detection, tracer=tracer)
    confidence_mod = _optional_confidence()
    narrate_mod = _optional_narrate() if narrate else None

    for group in group_findings(findings)[:max_cases]:
        entry = group[0]
        direction = direction_of(entry.test.relative_effect)
        mark = len(tracer.steps)

        with tracer.span(f"localize:{entry.metric}", kind="localizer") as span:
            span.what(f"Localizing the {direction} in {entry.metric}")
            span.why(
                "The detector says a metric moved; it does not say where. Localization removes "
                "each candidate in turn and asks whether the parent returns to expectation."
            )
            try:
                localization = localizer.localize(entry, direction=direction)
            except Exception as exc:  # noqa: BLE001 - one failed localization must not end the run
                log.warning("Localization failed for %s: %s", entry.metric, exc)
                span.result(f"failed: {exc}")
                continue
            accused = localization.accused
            span.result(
                f"accused {accused.segment.label()}" if accused else f"no candidate accused ({localization.mode})"
            )

        evidence = _finding_for_accused(localization, group, findings) or entry
        scored = _score(confidence_mod, localization, evidence, cfg, tracer)
        narration = _narrate(narrate_mod, localization, evidence, scored, cfg, tracer)

        case = build_case(
            evidence,
            localization,
            run_id=run_id,
            confidence=scored,
            narration=narration,
            trace_id=tracer.trace_id,
            steps=[{"case_id": "", **s.as_row()} for s in tracer.steps[mark:]],
        )
        result.cases.append(case)

    before = len(result.cases)
    result.cases = dedupe_cases(result.cases)
    result.cases.sort(key=lambda c: (-c.confidence_value, c.finding.test.p_value))
    if len(result.cases) < before:
        log.info("Collapsed %d case(s) that reached the same conclusion", before - len(result.cases))

    if persist:
        result.persisted = _persist(store, result, window, cfg, started, run_id)

    return result


def _score(module: Any | None, localization: Localization, finding: Finding, cfg: Config, tracer: Tracer) -> Any | None:
    if module is None:
        return None
    with tracer.span("confidence", kind="scoring") as span:
        span.what("Scoring the verdict across its graded components")
        span.why(
            "A verdict with no confidence attached forces an operator to treat a marginal "
            "result and an overwhelming one identically."
        )
        try:
            scored = module.score(localization, finding, cfg.confidence)
        except Exception as exc:  # noqa: BLE001
            log.warning("Confidence scoring failed: %s", exc)
            span.result(f"failed: {exc}")
            return None
        span.result(f"{scored.value:.2f} from {scored.components_scored}/{scored.components_total} components")
        return scored


def _narrate(
    module: Any | None,
    localization: Localization,
    finding: Finding,
    scored: Any | None,
    cfg: Config,
    tracer: Tracer,
) -> Any | None:
    if module is None:
        return None
    with tracer.span("narrate", kind="llm") as span:
        span.what("Phrasing the pre-computed claims as prose")
        span.why(
            "The model is given claim tuples and forbidden from computing. Every number it "
            "writes is checked against the bundle afterwards, and a draft containing an "
            "unsupported figure is discarded in favour of the template."
        )
        try:
            bundle = module.build_evidence(localization, finding, scored)
            narration = module.narrate(bundle, cfg.llm)
        except Exception as exc:  # noqa: BLE001 - prose is never worth losing a verdict over
            log.warning("Narration failed: %s", exc)
            span.result(f"failed: {exc}")
            return None
        span.result(f"{narration.source}, verified={narration.verified}")
        return narration


def _persist(
    store: CaseStore,
    result: InvestigationResult,
    window: Window,
    cfg: Config,
    started: datetime,
    run_id: str,
) -> bool:
    """Write everything out, treating a storage failure as a reporting failure and not a silent one."""
    ok = True
    for case in result.cases:
        for step in case.steps:
            step["case_id"] = case.case_id
        try:
            store.write_case(case)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to persist case %s: %s", case.case_id, exc)
            ok = False

    try:
        store.write_coverage(run_id, result.gaps, window)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to persist coverage ledger: %s", exc)
        ok = False

    try:
        store.close_run(
            run_id,
            cases_found=len(result.cases),
            status="complete" if ok else "partial",
            note="" if ok else "one or more writes failed; see logs",
            config_json=cfg.redacted_json(),
            git_sha=git_sha(),
            started_at=started,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to close run %s: %s", run_id, exc)
        ok = False

    return ok
