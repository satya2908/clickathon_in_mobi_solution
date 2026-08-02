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

import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .baseline import BaselineAudit, audit_baseline
from .config import Config
from .db import ClickHouse
from .detect import (
    CoverageGap,
    DetectionResult,
    Finding,
    apply_correction,
    detect_temporal,
    lattice_combos,
)
from .localize import Localization, Localizer
from .metrics import MetricRegistry
from .query import RollupReader, Window
from .schema import TOTAL_COMBO
from .store import Case, CaseStore, build_case, direction_of
from .structloc import SiblingLocalizer
from .structural import detect_structural
from .trace import NullTracer, Step, Tracer

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
    # Groups whose localization raised. Carried on the result rather than left in the log,
    # because the failure mode is silent by construction: a localization that dies produces no
    # case, and no case is indistinguishable from a clean window. A typo in a trace call once
    # emptied every fill_rate case from a run that still reported success.
    failures: list[str] = field(default_factory=list)
    baseline_audit: BaselineAudit | None = None

    @property
    def publishable(self) -> list[Case]:
        """Cases whose confidence actually cleared the bar, not merely all of them.

        This read `confidence_value >= 0.0`, which every case satisfies because the score is a
        weighted mean of non-negative components. It was a filter in name only. The threshold
        and the component-count rule already live on the confidence object, which is where the
        decision belongs; this defers to it rather than keeping a second, weaker copy.
        """
        out: list[Case] = []
        for case in self.cases:
            if case.verdict_kind != "localized":
                continue
            try:
                cleared = json.loads(case.confidence_json).get("publishable")
            except (TypeError, ValueError):
                cleared = False
            if cleared:
                out.append(case)
        return out

    @property
    def temporal_disabled(self) -> bool:
        return self.baseline_audit is not None and not self.baseline_audit.trustworthy

    def summary(self) -> str:
        out = (
            f"run {self.run_id}: {self.cells_tested:,} cells tested across "
            f"{len(self.metrics_scanned)} metrics, {self.findings_after_correction} findings "
            f"survived correction, {len(self.cases)} case(s), {len(self.gaps)} coverage gap(s)"
        )
        if self.temporal_disabled:
            out += ", AGGREGATE-ONLY HISTORY (segment baseline rejected)"
        if self.failures:
            out += f", {len(self.failures)} localization(s) FAILED"
        return out


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

    The metric has to match as well as the segment. ``everything`` spans the whole sweep, so a
    segment that moved in two metrics has a finding in each, and matching on segment alone let a
    fill_rate localization quote the requests finding for the same cell -- producing a case
    labelled ``requests`` whose observed value was 0.599, because the numbers come from the
    localization and only the name came from the finding. A different metric's test is not
    evidence about this claim, so it is never a candidate.
    """
    accused = localization.accused
    if accused is None:
        return None
    for pool in (group, everything):
        matches = [f for f in pool if f.segment == accused.segment and f.metric == localization.metric]
        if matches:
            matches.sort(key=lambda f: (f.detector != "temporal", f.test.p_value))
            return matches[0]
    # Nothing in the sweep tested this exact cell, which is common: localization reasons over
    # the whole lattice while the detector only reports what tripped. The localizer re-tests the
    # accused cell against its own history for exactly this case, and that result is evidence
    # about this claim in a way that a neighbouring cell's p-value is not.
    return localization.accused_finding


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
        group.sort(key=_strength)
        groups.append(group)
    groups.sort(key=lambda g: _strength(g[0]))
    return groups


def _strength(finding: Finding) -> tuple[float, float, float]:
    """Ordering key: strongest evidence first.

    ``p_value`` alone is not enough. A median-polish residual eight standard errors out
    underflows to exactly 0.0, and so does one at thirty, so every structural finding in a run
    ties at the head of the list and the group that gets investigated is whichever one happened
    to sort first. Falling through to the standardised residual breaks that tie by how far out
    the cell actually sits, which is the thing p was standing in for before it saturated.
    """
    return (finding.test.p_value, -abs(finding.test.z), -abs(finding.test.relative_effect))


def for_metric(steps: list[Step], metric: str) -> list[Step]:
    """The part of the run-level prefix that concerns one metric.

    A sweep over ten metrics emits a detection span per metric per lattice combination, and
    copying all of them onto every case produced four thousand rows for nine cases -- a
    three-megabyte page, most of it a fill_rate case carrying the CTR scan. The other metrics'
    spans are not evidence for this verdict; they are what else happened to run at the time.

    Named spans follow ``stage:metric:combo``, so anything with fewer than two segments --
    ``investigate``, ``detect``, ``correct`` -- is structural and always kept, which also keeps
    every retained child's parent present in the result.
    """
    keep = []
    for step in steps:
        parts = step.name.split(":")
        if len(parts) < 2 or parts[1] == metric:
            keep.append(step)
    return keep


def detect_all(
    reader: RollupReader,
    registry: MetricRegistry,
    cfg: Config,
    window: Window,
    *,
    metrics: list[str] | None = None,
    tracer: Tracer | None = None,
    structural: bool = True,
    temporal_enabled: bool = True,
    temporal_combos: list[str] | None = None,
) -> tuple[DetectionResult, DetectionResult]:
    """Run both detectors over every requested metric, returning them unmixed.

    They are kept apart because only one of them can be FDR-corrected honestly; see the module
    docstring. Callers that want a single list should correct the temporal result first and then
    concatenate, which is what ``investigate`` does.

    ``temporal_enabled`` switches history off entirely. ``temporal_combos`` narrows what it is
    allowed to compare instead, and the audit uses it to keep the platform aggregate when it has
    rejected segment-level history: reissuing which entities carry which attribute rearranges
    every segment and cannot move a total, so the total's own history stays comparable when
    nothing below it does. Dropping it along with the rest would leave a platform-wide movement
    with no detector able to see it, since the structural comparison is between siblings and a
    total has none.
    """
    names = metrics or list(registry.metrics)
    temporal = DetectionResult()
    struct = DetectionResult()

    # One read for the whole sweep. Each metric legally sees a different set of combinations,
    # but the counters behind them are the same five columns, so the union is fetched once and
    # every metric is scanned against it. Under per_combo this returns 0 and the detectors
    # fall through to querying combination by combination.
    wanted: list[str] = []
    for name in names:
        wanted.extend(lattice_combos(registry, registry.metric(name), window.grain))
    cells = reader.prefetch_lattice(wanted, window, cfg.detection.baseline_weeks)
    if cells:
        log.info(
            "Prefetched %s cells across %s combos in one query", f"{cells:,}", len(set(wanted))
        )

    for name in names:
        if temporal_enabled:
            result = detect_temporal(
                reader, registry, cfg.detection, name, window,
                combos=temporal_combos, tracer=tracer, correct=False,
            )
            # extend() rather than extending the lists by hand, because it also carries
            # tested_cells. Benjamini-Hochberg sizes its family from that count, and copying
            # only the findings across left it at zero -- so the family collapsed to the number
            # of findings and every threshold was computed against a denominator far smaller
            # than the number of tests actually run. Latent while every tested cell yields a
            # finding, and silently permissive the moment one does not.
            temporal.extend(result)

        if not structural:
            continue
        try:
            found = detect_structural(reader, registry, cfg.detection, name, window, tracer=tracer)
        except Exception as exc:  # noqa: BLE001 - one metric's failure must not end the scan
            log.warning("Structural scan failed for %s: %s", name, exc)
            continue
        struct.extend(found)

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
    """Investigate one window under a single root span.

    The root exists so the run is one trace rather than several. Each stage used to open its own
    top-level span, and a top-level span starts a new trace: one run produced 56 traces across
    287 spans, with detect, correct, localize, confidence and narrate all unconnected. A case
    stores one trace id to deep-link a reader into HyperDX, so that link opened whichever
    fragment the case happened to capture instead of the investigation that produced it.

    Nesting everything here also makes the trace match the mental model the case file describes:
    one investigation, with the stages inside it in the order they ran.
    """
    tracer = tracer or NullTracer()
    with tracer.span("investigate", kind="pipeline") as span:
        span.what(f"Investigating {window.label()} at {window.grain} grain")
        span.why(
            "One root span per run, so every stage below shares a trace id and the case can "
            "link a reader to the whole investigation rather than to one stage of it."
        )
        result = _investigate(
            cfg, ch, registry, window,
            metrics=metrics, tracer=tracer, persist=persist,
            narrate=narrate, max_cases=max_cases,
        )
        span.result(
            f"{len(result.cases)} case(s) from {result.findings_after_correction} finding(s) "
            f"that survived correction"
        )
        return result


def _investigate(
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
    reader = RollupReader(ch, read_mode=cfg.clickhouse.read_mode)
    store = CaseStore(ch)
    result = InvestigationResult(run_id=run_id, window=window)
    result.metrics_scanned = metrics or list(registry.metrics)

    if persist:
        try:
            store.open_run(run_id, config_json=cfg.redacted_json(), git_sha=git_sha())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open run row: %s", exc)

    if cfg.detection.baseline_audit_enabled:
        with tracer.span("audit", kind="statistics") as span:
            span.what("Checked whether the baseline still describes this population")
            span.why(
                "A baseline drawn from a population that has since changed produces confident, "
                "internally consistent, wrong answers, and nothing downstream can detect that. "
                "A calibrated baseline disagrees with a few percent of a recent window; one "
                "describing the wrong population disagrees with most of it."
            )
            result.baseline_audit = audit_baseline(
                reader, registry, cfg, window, metrics=result.metrics_scanned
            )
            span.result(result.baseline_audit.headline)

    use_temporal = result.baseline_audit is None or result.baseline_audit.trustworthy
    if not use_temporal:
        log.error("%s", result.baseline_audit.detail)

    with tracer.span("detect", kind="detector") as span:
        span.what(f"Scanned {len(result.metrics_scanned)} metric(s) over {window.label()} at {window.grain}")
        span.why(
            "Every cell in the lattice is compared against its own history and against its "
            "siblings, because an incident confined to one cell is invisible in the total."
            if use_temporal
            else "Segment-level history failed its audit, so segments are compared only against "
            "their siblings in the same window, which needs no baseline. The platform aggregate "
            "keeps its own history: relabelling entities rearranges segments and cannot move a "
            "total, and a total has no siblings to be compared against instead."
        )
        temporal, struct = detect_all(
            reader, registry, cfg, window, metrics=metrics, tracer=tracer,
            structural=True,
            temporal_combos=None if use_temporal else [TOTAL_COMBO],
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
    siblings = SiblingLocalizer(reader, registry, cfg.localization, cfg.detection, tracer=tracer)
    confidence_mod = _optional_confidence()
    narrate_mod = _optional_narrate() if narrate else None

    # Everything traced so far belongs to the run rather than to any one case: the root span, the
    # lattice sweep, and the correction. Each case copies the part of it that concerns its own
    # metric, so that reading one case_id back gives the whole investigation rather than its last
    # three steps. Without this a stored case showed only localize, confidence and narrate -- the
    # stages that justify the verdict, the sweep that found it and the correction that survived
    # it, were absent.
    #
    # Copied rather than normalised into a run-level table, so one query by case_id returns a
    # self-contained tree whose parent references all resolve inside it.
    shared = list(tracer.steps)

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
                # Every counterfactual the historical localizer applies -- sufficiency,
                # minimality, maximality, holdout -- asks whether something returned to
                # *expectation*, and expectation comes from the history this run has already
                # rejected. Naming a segment on that basis would launder a baseline we do not
                # believe into a verdict we would defend. So when the audit fails, the question
                # is asked against siblings in the same window instead, which needs no history
                # and so is untouched by the rejection.
                chosen = localizer if use_temporal else siblings
                localization = chosen.localize(entry, direction=direction)
            except Exception as exc:  # noqa: BLE001 - one failed localization must not end the run
                # Logged at error, recorded on the result, and reported by the run's status.
                # A warning was not enough: the only other trace of this is a case that never
                # appears, which reads exactly like a window with nothing wrong in it.
                log.error("Localization failed for %s: %s", entry.metric, exc, exc_info=True)
                span.result(f"failed: {exc}")
                result.failures.append(f"{entry.metric}: {exc}")
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
            cells_tested=result.cells_tested,
            steps=[
                {"case_id": "", **s.as_row()}
                for s in (*for_metric(shared, entry.metric), *tracer.steps[mark:])
            ],
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
        store.write_cases(result.cases)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to persist cases for run %s: %s", run_id, exc)
        ok = False

    try:
        store.write_coverage(run_id, result.gaps, window)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to persist coverage ledger: %s", exc)
        ok = False

    try:
        # From the in-memory datetime, which keeps microseconds, rather than from the difference
        # of the two stored timestamps, which are truncated to whole seconds.
        elapsed_ms = max(0, round((datetime.now(UTC) - started).total_seconds() * 1000))
        # A run that could not localize part of what it found is not a complete run, whatever
        # the writes did. Recorded in the status so the console does not present it as a clean
        # window, which is what an empty case list otherwise looks like.
        notes = []
        if not ok:
            notes.append("one or more writes failed; see logs")
        if result.failures:
            notes.append(f"{len(result.failures)} localization(s) failed: "
                         + "; ".join(result.failures[:3]))
        store.close_run(
            run_id,
            cases_found=len(result.cases),
            status="complete" if ok and not result.failures else "partial",
            note=" | ".join(notes),
            config_json=cfg.redacted_json(),
            git_sha=git_sha(),
            started_at=started,
            duration_ms=elapsed_ms,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to close run %s: %s", run_id, exc)
        ok = False

    return ok
