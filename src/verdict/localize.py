"""Localization: telling a cause apart from a passenger.

Ranking segments by how far they moved does not work, and the reason is worth being precise
about. When Android 15 fill rate collapses, every device model that skews Android 15 also
drops. Galaxy S23 shows a large, statistically significant decline and is entirely innocent.
Ranking cannot separate them because from above they look the same.

What separates them is a counterfactual. Remove the accused segment from the population and
recompute the parent metric. If the parent returns to normal, the accused explains it. If the
parent barely moves, the accused was a passenger. Removing Android 15 restores global fill
rate; removing Galaxy S23 leaves almost all of the drop in place.

Sufficiency alone is not enough, because it is monotone in breadth: a segment large enough to
contain the real cause always explains at least as much as the cause itself. "All traffic"
explains everything perfectly and says nothing. Two more tests bound the answer from both
sides.

  * **Minimality** asks whether the candidate is too broad. Take away its guiltiest child. If
    the candidate stops deviating, the child was the real answer and the candidate was
    carrying it.
  * **Maximality** asks whether the candidate is too narrow. If its siblings under the same
    parent moved the same way, the honest answer is the parent, and naming one child sends an
    operator to inspect a device model when the whole platform is affected.

Every test returns pass, fail, or **unknown**, and the third state is load-bearing. Minimality
on a two-dimensional candidate would need a three-dimensional cell, which the lattice does not
carry. Reporting that as a pass would claim a check was performed that never ran.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .config import DetectionConfig, LocalizationConfig
from .detect import Finding
from .metrics import Metric, MetricRegistry
from .query import Counters, RollupReader, Segment, Window, subtract
from .schema import LATTICE_DEPTH, TOTAL_COMBO
from .stats import median
from .trace import Tracer

log = logging.getLogger(__name__)

State = Literal["pass", "fail", "unknown"]

# Below this, a deviation is too small for a ratio of deviations to mean anything: the
# denominator of the sufficiency fraction is mostly rounding noise.
_MIN_TESTABLE_DEVIATION = 1e-9


@dataclass(frozen=True)
class Check:
    """One localization test on one candidate."""

    name: str
    state: State
    score: float | None
    detail: str

    @property
    def passed(self) -> bool:
        return self.state == "pass"


@dataclass
class Candidate:
    segment: Segment
    observed: Counters
    expected: Counters
    observed_value: float | None
    expected_value: float | None
    checks: dict[str, Check] = field(default_factory=dict)
    predicted_if_innocent: float | None = None
    exoneration_residual: float | None = None
    status: str = "considered"
    reason: str = ""

    @property
    def deviation(self) -> float:
        if self.observed_value is None or self.expected_value is None:
            return 0.0
        return self.observed_value - self.expected_value

    @property
    def relative_effect(self) -> float:
        if not self.expected_value:
            return 0.0
        return self.deviation / self.expected_value

    @property
    def sufficiency(self) -> float:
        check = self.checks.get("sufficiency")
        return check.score if check and check.score is not None else 0.0

    def check_state(self, name: str) -> State:
        check = self.checks.get(name)
        return check.state if check else "unknown"


@dataclass
class Localization:
    """The outcome of localizing one finding."""

    metric: str
    window: Window
    parent: Segment
    parent_observed: float | None
    parent_expected: float | None
    parent_deviation: float
    accused: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)
    mode: str = "explain_away"
    note: str = ""

    @property
    def cleared(self) -> list[Candidate]:
        return [c for c in self.candidates if c.status == "cleared"]

    @property
    def runners_up(self) -> list[Candidate]:
        return [c for c in self.candidates if c.status in {"too_broad", "too_narrow", "partial"}]


class HistoryCache:
    """Per-combo history, fetched once per investigation.

    Localization asks about the same combos repeatedly -- once to enumerate candidates, again
    for every minimality and maximality test. Refetching would be slow and, worse, would read
    from different snapshots of a table that is still merging, so two numbers in the same case
    file could come from different states of the data.
    """

    def __init__(self, reader: RollupReader, window: Window, weeks: int) -> None:
        self.reader = reader
        self.window = window
        self.weeks = weeks
        self._cache: dict[str, dict[Segment, list[Counters]]] = {}
        # Set once per localization from the parent's history, then used for every segment in
        # it. Holding it here rather than passing it down each call is what makes it impossible
        # for two segments in the same comparison to be averaged over different weeks.
        self.mask: tuple[int, ...] | None = None

    def combo(self, combo: str) -> dict[Segment, list[Counters]]:
        if combo not in self._cache:
            self._cache[combo] = self.reader.slice_with_history(combo, self.window, self.weeks)
        return self._cache[combo]

    def adopt_mask(self, parent_history: list[Counters], metric: Metric, *, trim: bool) -> None:
        self.mask = trim_mask(parent_history, metric, trim=trim)

    def expected(self, weeks: list[Counters]) -> Counters:
        """Expectation for one segment, over the localization's shared set of baseline weeks."""
        return expected_counters(weeks[1:], self.mask)

    def segment(self, segment: Segment) -> tuple[Counters, Counters] | None:
        """Observed and expected counters for one segment."""
        cells = self.combo(segment.combo)
        weeks = cells.get(segment)
        if weeks is None:
            return None
        return weeks[0], self.expected(weeks)

    def siblings(self, segment: Segment, dimension: str) -> dict[Segment, list[Counters]]:
        """Cells sharing every key with ``segment`` except the value of ``dimension``."""
        fixed = {k: v for k, v in segment.keys if k != dimension}
        out = {}
        for candidate, weeks in self.combo(segment.combo).items():
            keys = candidate.as_dict()
            if all(keys.get(k) == v for k, v in fixed.items()) and candidate != segment:
                out[candidate] = weeks
        return out


def trim_mask(history: list[Counters], metric: Metric, *, trim: bool = True) -> tuple[int, ...]:
    """Which baseline weeks to average, as indices into ``history``.

    Chosen once per localization and reused for every segment in it. Choosing per segment is
    what made the counterfactuals unsound: if the parent drops week two and a candidate drops
    week four, then `parent_expected - candidate_expected` subtracts quantities averaged over
    different weeks, and nothing makes the candidate's requests smaller than the parent's. The
    result is not nested, so the subtraction produces a rate outside [0, 1] -- a confident,
    precise, meaningless verdict.

    Trimming discards the single week furthest from the median rate. It exists so that one
    prior incident in the baseline window does not define "normal" as the incident, and one
    week is the most that can be dropped from four while leaving a median worth the name.
    """
    if not history:
        return ()

    den_field = metric.denominator_field or metric.numerator_field
    usable = [i for i, c in enumerate(history) if getattr(c, den_field) > 0]
    if not trim or len(usable) < 3:
        return tuple(usable)

    # Deliberately the metric under investigation, not fill rate. A CTR case trimmed on fill
    # rate discards whichever week was odd for a quantity nobody asked about, while leaving in
    # the week that actually distorts the CTR baseline.
    rates = [(history[i].value(metric) or 0.0) for i in usable]
    centre = median(rates)
    worst = max(range(len(usable)), key=lambda j: abs(rates[j] - centre))
    return tuple(i for j, i in enumerate(usable) if j != worst)


def expected_counters(
    history: list[Counters], mask: Sequence[int] | None = None, *, trim: bool = True
) -> Counters:
    """Counters the segment would have carried had nothing changed.

    Each counter is pooled over the weeks in ``mask`` and divided by how many there were,
    giving a per-window expectation directly comparable to the observation. Every counter is
    averaged over the same weeks, so the resulting tuple stays internally consistent -- taking
    requests from one set of weeks and fills from another would produce an expected fill rate
    that no week ever had.

    ``mask`` comes from `trim_mask`, computed once for the whole localization. Passing None
    falls back to every week with a non-zero request count, which is only correct when the
    result will not be subtracted from another segment's expectation.
    """
    if mask is None:
        keep = [c for c in history if c.requests > 0]
    else:
        keep = [history[i] for i in mask if i < len(history)]
    if not keep:
        return Counters()

    n = len(keep)
    total = Counters()
    for c in keep:
        total = total + c
    return Counters(
        requests=int(round(total.requests / n)),
        fills=int(round(total.fills / n)),
        impressions=int(round(total.impressions / n)),
        clicks=int(round(total.clicks / n)),
        revenue=total.revenue / n,
    )


def _value(counters: Counters, metric: Metric) -> float | None:
    return counters.value(metric)


def sufficiency_check(
    metric: Metric,
    parent_obs: Counters,
    parent_exp: Counters,
    candidate: Candidate,
    threshold: float,
) -> Check:
    """Remove the candidate and see whether the parent returns to normal.

    The counterfactual is built from counters rather than by re-querying, so the removal is
    exact and the denominator of the recomputed metric is guaranteed to match the one the
    original deviation was measured against.
    """
    original = (_value(parent_obs, metric) or 0.0) - (_value(parent_exp, metric) or 0.0)
    if abs(original) < _MIN_TESTABLE_DEVIATION:
        return Check(
            "sufficiency",
            "unknown",
            None,
            "The parent metric did not move, so there is no deviation for this candidate to "
            "explain. Common when two segments moved in opposite directions and cancelled.",
        )

    remainder_obs = subtract(parent_obs, candidate.observed)
    remainder_exp = subtract(parent_exp, candidate.expected)
    if remainder_obs is None or remainder_exp is None:
        return Check(
            "sufficiency",
            "unknown",
            None,
            "This candidate is not contained in the parent for every counter, so removing it "
            "is not a counterfactual the lattice can express.",
        )

    residual_obs = _value(remainder_obs, metric)
    residual_exp = _value(remainder_exp, metric)
    if residual_obs is None or residual_exp is None:
        return Check(
            "sufficiency",
            "unknown",
            None,
            "Removing this candidate empties the parent population, so the counterfactual "
            "has no denominator.",
        )

    residual = residual_obs - residual_exp
    score = max(0.0, 1.0 - abs(residual) / abs(original))
    detail = (
        f"Parent moved {original:+.6g}. With {candidate.segment.label()} removed it moves "
        f"{residual:+.6g}, so {score:.0%} of the change is accounted for by this segment."
    )
    return Check("sufficiency", "pass" if score >= threshold else "fail", score, detail)


def minimality_check(
    metric: Metric,
    candidate: Candidate,
    history: HistoryCache,
    registry: MetricRegistry,
    threshold: float,
) -> Check:
    """Is the candidate carrying a smaller segment that is the real answer?

    Take the candidate's guiltiest child away. If what remains still deviates, the candidate
    is genuinely the right level. If the deviation vanishes, the child was the cause all along
    and reporting the parent would send an operator to look at far too much traffic.
    """
    if candidate.segment.is_total:
        return Check(
            "minimality", "fail", 0.0,
            "The whole population always explains itself. Naming it identifies nothing.",
        )

    original = candidate.deviation
    if abs(original) < _MIN_TESTABLE_DEVIATION:
        return Check("minimality", "unknown", None, "Candidate shows no deviation to attribute.")

    if candidate.segment.depth >= 2:
        return Check(
            "minimality", "unknown", None,
            "Testing this would need a three-dimensional cell. The rollup lattice carries one "
            "and two-dimensional combinations only, so no child of this candidate exists to "
            "remove. Recorded as untested rather than passed.",
        )

    dimension = candidate.segment.dimensions[0]
    others = [d for d in registry.valid_dimensions(metric) if d != dimension]

    # Seeded above any achievable residual rather than at the candidate's own deviation. Seeding
    # at the deviation would silently discard the case where no child reduces it -- which is not
    # an absence of evidence but the strongest possible pass, meaning the movement is spread
    # evenly and this really is the right level to report.
    worst_child: Segment | None = None
    worst_residual = float("inf")
    tested = 0
    for other in others:
        combo = "|".join(sorted((dimension, other)))
        for child, weeks in history.combo(combo).items():
            if child.as_dict().get(dimension) != candidate.segment.as_dict()[dimension]:
                continue
            child_obs, child_exp = weeks[0], history.expected(weeks)
            without_obs = subtract(candidate.observed, child_obs)
            without_exp = subtract(candidate.expected, child_exp)
            if without_obs is None or without_exp is None:
                continue
            remainder_obs = _value(without_obs, metric)
            remainder_exp = _value(without_exp, metric)
            if remainder_obs is None or remainder_exp is None:
                continue
            tested += 1
            residual = abs(remainder_obs - remainder_exp)
            if residual < worst_residual:
                worst_residual = residual
                worst_child = child

    if worst_child is None or not tested:
        return Check(
            "minimality", "unknown", None,
            "No child cell had enough traffic to test whether it carries the deviation.",
        )

    # Capped at 1.0 because removing a child can leave a remainder that deviates by *more* than
    # the whole did, when the child was moving the other way. That is still a pass, and a score
    # above 1 would misrepresent it as more than complete.
    score = min(worst_residual / abs(original), 1.0)
    if score >= threshold:
        detail = (
            f"Removing the guiltiest child ({worst_child.label()}) still leaves "
            f"{score:.0%} of the deviation, so the cause is spread across this segment "
            "rather than concentrated in one part of it."
        )
        return Check("minimality", "pass", score, detail)

    detail = (
        f"Removing {worst_child.label()} collapses the deviation to {score:.0%} of its "
        "original size. This candidate is too broad; the child is the real answer."
    )
    return Check("minimality", "fail", score, detail)


def maximality_check(
    metric: Metric,
    candidate: Candidate,
    history: HistoryCache,
    threshold: float,
) -> Check:
    """Is the candidate an arbitrarily narrow slice of something wider?

    If every sibling under the same parent moved the same way and by a similar amount, the
    honest answer is the parent. Naming one child would send an operator to investigate a
    single device model when the entire platform is affected.
    """
    if candidate.segment.is_total:
        return Check("maximality", "pass", 1.0, "Nothing is wider than the whole population.")

    original = candidate.relative_effect
    if abs(original) < 1e-9:
        return Check("maximality", "unknown", None, "Candidate shows no deviation to compare.")

    # `worst_share` starts below zero so that a share of exactly 0.0 -- no sibling followed the
    # candidate, the strongest possible pass -- still registers as a dimension that was
    # evaluated. Starting at 0.0 with a strict comparison would report the cleanest result in
    # the whole check as "could not be tested".
    worst_share = -1.0
    worst_dimension = ""
    worst_count = 0

    for dimension in candidate.segment.dimensions:
        siblings = history.siblings(candidate.segment, dimension)
        # One sibling can only ever answer 0% or 100%, which is not evidence about how wide the
        # movement is. Every dimension in the real lattice has at least three values.
        if len(siblings) < 2:
            continue

        agreeing = 0
        considered = 0
        for weeks in siblings.values():
            sib_obs, sib_exp = weeks[0], history.expected(weeks)
            obs_v, exp_v = _value(sib_obs, metric), _value(sib_exp, metric)
            if obs_v is None or exp_v is None or not exp_v:
                continue
            considered += 1
            sibling_effect = (obs_v - exp_v) / exp_v
            # Same direction and at least half the magnitude counts as sharing the movement.
            if sibling_effect * original > 0 and abs(sibling_effect) >= 0.5 * abs(original):
                agreeing += 1

        if considered and agreeing / considered > worst_share:
            worst_share = agreeing / considered
            worst_dimension = dimension
            worst_count = considered

    if not worst_dimension:
        return Check(
            "maximality", "unknown", None,
            "No sibling cell had enough traffic to test whether the movement is wider than "
            "this candidate.",
        )

    if worst_share >= threshold:
        detail = (
            f"{worst_share:.0%} of the {worst_count} sibling values of {worst_dimension} "
            "moved the same way by a comparable amount. The movement is wider than this "
            "candidate, so naming it would be arbitrarily narrow."
        )
        return Check("maximality", "fail", 1.0 - worst_share, detail)

    detail = (
        f"Only {worst_share:.0%} of the {worst_count} sibling values of {worst_dimension} "
        "moved with it, so the deviation really is specific to this candidate."
    )
    return Check("maximality", "pass", 1.0 - worst_share, detail)


def holdout_check(
    reader: RollupReader,
    metric: Metric,
    candidate: Candidate,
    window: Window,
    weeks: int,
) -> Check:
    """Split the window and see whether the verdict reproduces on the half it was not chosen on.

    A candidate selected because it deviated across the whole window has had every opportunity
    to be selected by chance. If the same segment deviates in the same direction on the second
    half alone, that is a fresh observation rather than a restatement of the one that picked
    it. Bleed-through from a genuine culprit tends to weaken; a real cause reproduces.
    """
    midpoint = window.start + window.duration / 2
    first = Window(window.start, midpoint, window.grain)
    second = Window(midpoint, window.end, window.grain)
    if first.duration.total_seconds() <= 0 or second.duration.total_seconds() <= 0:
        return Check("holdout", "unknown", None, "Window too short to split.")

    def effect(part: Window) -> float | None:
        observed = reader.segment(candidate.segment, part)
        history = [
            reader.segment(candidate.segment, part.shifted(w)) for w in range(1, weeks + 1)
        ]
        # Trimmed on its own history rather than the localization's shared mask. This compares
        # one segment against itself over half-windows, so no cross-segment subtraction happens
        # and nesting is not at stake; and a week that was atypical across the whole window is
        # not necessarily atypical in each half.
        exp = expected_counters(history, trim_mask(history, metric, trim=True))
        obs_v, exp_v = _value(observed, metric), _value(exp, metric)
        if obs_v is None or exp_v is None or not exp_v:
            return None
        return (obs_v - exp_v) / exp_v

    a, b = effect(first), effect(second)
    if a is None or b is None:
        return Check("holdout", "unknown", None, "One half had no comparable traffic.")

    if a * b <= 0:
        return Check(
            "holdout", "fail", 0.0,
            f"The first half moved {a:+.1%} and the second {b:+.1%}. The two halves disagree "
            "on direction, which is what an artefact of window choice looks like.",
        )

    ratio = min(abs(a), abs(b)) / max(abs(a), abs(b))
    detail = (
        f"First half {a:+.1%}, second half {b:+.1%}. The effect reproduces on the half it "
        "was not selected on."
    )
    return Check("holdout", "pass" if ratio >= 0.5 else "fail", ratio, detail)


def exonerate(
    metric: Metric,
    candidate: Candidate,
    accused: Candidate,
    overlap_share: float,
) -> tuple[float | None, float | None]:
    """Predict what an innocent bystander should read, then compare with what it does read.

    If the accused explains everything, any other segment's movement must be exactly the
    dilution of the accused's movement by however much of that segment the accused occupies:

        predicted = share * accused_rate_now + (1 - share) * segment_rate_before

    Publishing predicted against observed turns clearing a candidate into a falsifiable claim.
    A residual near zero means the movement is fully accounted for. A residual that refuses to
    close is a second cause, and saying so is more useful than silently omitting the segment.
    """
    accused_now = _value(accused.observed, metric)
    baseline = _value(candidate.expected, metric)
    observed = _value(candidate.observed, metric)
    if accused_now is None or baseline is None or observed is None:
        return None, None

    predicted = overlap_share * accused_now + (1.0 - overlap_share) * baseline
    return predicted, observed - predicted


class Localizer:
    def __init__(
        self,
        reader: RollupReader,
        registry: MetricRegistry,
        localization: LocalizationConfig,
        detection: DetectionConfig,
        tracer: Tracer | None = None,
    ) -> None:
        self.reader = reader
        self.registry = registry
        self.cfg = localization
        self.detection = detection
        self.tracer = tracer

    def localize(self, finding: Finding) -> Localization:
        metric = self.registry.metric(finding.metric)
        window = finding.window
        history = HistoryCache(self.reader, window, self.detection.baseline_weeks)

        parent = Segment.total()
        parent_cells = history.combo(TOTAL_COMBO)
        parent_weeks = parent_cells.get(parent)
        if parent_weeks is None:
            return Localization(
                metric=metric.name, window=window, parent=parent,
                parent_observed=None, parent_expected=None, parent_deviation=0.0,
                accused=None, mode="no_data",
                note="No total-level rollup rows exist for this window.",
            )

        # The trim decision is made here, once, from the parent's history and the metric under
        # investigation, and every expectation in this localization then uses it. Deciding it
        # per segment lets the parent and a candidate average over different weeks, which
        # breaks the containment that `parent_expected - candidate_expected` relies on.
        parent_obs = parent_weeks[0]
        history.adopt_mask(parent_weeks[1:], metric, trim=self.detection.trim_extremes)
        parent_exp = history.expected(parent_weeks)
        parent_obs_v = _value(parent_obs, metric)
        parent_exp_v = _value(parent_exp, metric)
        parent_dev = (parent_obs_v or 0.0) - (parent_exp_v or 0.0)

        mode = "explain_away"
        note = ""
        if abs(parent_dev) < _MIN_TESTABLE_DEVIATION:
            mode = "structural_only"
            note = (
                "The parent metric did not move, so there is nothing for a counterfactual to "
                "restore. The candidate is judged on its own deviation and on whether its "
                "siblings share it."
            )

        candidates = self._build_candidates(metric, finding, history)
        for candidate in candidates:
            candidate.checks["sufficiency"] = sufficiency_check(
                metric, parent_obs, parent_exp, candidate, self.cfg.sufficiency_threshold
            )
            candidate.checks["minimality"] = minimality_check(
                metric, candidate, history, self.registry, self.cfg.minimality_threshold
            )
            candidate.checks["maximality"] = maximality_check(
                metric, candidate, history, self.cfg.maximality_threshold
            )

        accused = self._choose(candidates, mode)

        if accused is not None and self.cfg.holdout_enabled:
            accused.checks["holdout"] = holdout_check(
                self.reader, metric, accused, window, self.detection.baseline_weeks
            )

        if accused is not None:
            self._build_ledger(metric, accused, candidates, history)

        if self.tracer is not None:
            self._trace(metric, finding, parent_dev, candidates, accused, mode)

        return Localization(
            metric=metric.name,
            window=window,
            parent=parent,
            parent_observed=parent_obs_v,
            parent_expected=parent_exp_v,
            parent_deviation=parent_dev,
            accused=accused,
            candidates=candidates,
            mode=mode,
            note=note,
        )

    def _build_candidates(
        self, metric: Metric, finding: Finding, history: HistoryCache
    ) -> list[Candidate]:
        """Every cell worth testing, ranked by how much of the movement it could carry.

        Capped at ``max_candidates`` by absolute contribution, not by relative effect. A tiny
        segment that halved is a large relative move and cannot account for a change in the
        total, while a large segment that slipped a few percent can.
        """
        # Capped at what this grain actually materializes. Enumerating deeper would return
        # nothing for the missing combos and quietly shrink the candidate set: a one-way segment
        # that should have failed minimality against its two-way child would be accused instead,
        # with no gap recorded anywhere to show the child was never considered.
        depth = min(self.cfg.max_depth, LATTICE_DEPTH.get(finding.window.grain, 2))
        legal = sorted(self.registry.valid_dimensions(metric))
        combos: list[str] = list(legal) if depth >= 1 else []
        if depth >= 2:
            for i, a in enumerate(legal):
                for b in legal[i + 1 :]:
                    combos.append(f"{a}|{b}")

        scored: list[tuple[float, Candidate]] = []
        for combo in combos:
            for segment, weeks in history.combo(combo).items():
                if segment.depth > self.cfg.max_depth:
                    continue
                observed = weeks[0]
                expected = history.expected(weeks)
                obs_v, exp_v = _value(observed, metric), _value(expected, metric)
                if obs_v is None or exp_v is None:
                    continue

                # Absolute contribution: how much of the parent's numerator this segment's
                # movement could account for.
                weight = observed.denominator(metric) if metric.is_ratio else 1.0
                contribution = abs(obs_v - exp_v) * (weight or 1.0)
                scored.append(
                    (contribution, Candidate(segment, observed, expected, obs_v, exp_v))
                )

        # Always keep the segment the detector actually flagged, even if its absolute
        # contribution ranks it out. A compensating pair has small net contribution by
        # construction, and dropping it here would discard the finding that started this.
        scored.sort(key=lambda pair: -pair[0])
        kept = [c for _, c in scored[: self.cfg.max_candidates]]
        if not finding.segment.is_total and all(c.segment != finding.segment for c in kept):
            for _, candidate in scored:
                if candidate.segment == finding.segment:
                    kept.append(candidate)
                    break
        return kept

    def _choose(self, candidates: list[Candidate], mode: str) -> Candidate | None:
        """Pick the narrowest candidate that survives every test it was possible to run.

        Ordering is by sufficiency, then by depth descending. The tie-break matters: when a
        two-dimensional cell and the one-dimensional segment containing it explain the same
        amount, the narrower one is the more useful answer, and minimality has already
        eliminated it if it was too narrow to deserve that.
        """
        viable: list[Candidate] = []
        for candidate in candidates:
            suff = candidate.check_state("sufficiency")
            minim = candidate.check_state("minimality")
            maxim = candidate.check_state("maximality")

            if minim == "fail":
                candidate.status = "too_broad"
                candidate.reason = candidate.checks["minimality"].detail
                continue
            if maxim == "fail":
                candidate.status = "too_narrow"
                candidate.reason = candidate.checks["maximality"].detail
                continue
            if mode == "explain_away" and suff == "fail":
                candidate.status = "partial"
                candidate.reason = candidate.checks["sufficiency"].detail
                continue
            viable.append(candidate)

        if not viable:
            return None

        viable.sort(key=lambda c: (-c.sufficiency, -c.segment.depth))
        accused = viable[0]
        accused.status = "accused"
        for other in viable[1:]:
            other.status = "partial"
            other.reason = "Explains less of the movement than the accused segment."
        return accused

    def _build_ledger(
        self,
        metric: Metric,
        accused: Candidate,
        candidates: list[Candidate],
        history: HistoryCache,
    ) -> None:
        """Publish predicted against observed for every candidate that was not accused."""
        for candidate in candidates:
            if candidate is accused:
                continue
            share = self._overlap(metric, candidate, accused, history)
            if share is None:
                candidate.reason = candidate.reason or (
                    "Could not be cleared quantitatively: no rollup cell covers the overlap "
                    "between this segment and the accused."
                )
                continue

            predicted, residual = exonerate(metric, candidate, accused, share)
            candidate.predicted_if_innocent = predicted
            candidate.exoneration_residual = residual

            if residual is None:
                continue
            if abs(residual) <= self.cfg.exoneration_residual_tolerance:
                if candidate.status == "considered":
                    candidate.status = "cleared"
                candidate.reason = (
                    f"Predicted {predicted:.6g} if the accused explains everything; observed "
                    f"{candidate.observed_value:.6g}. Residual {residual:+.6g} is within "
                    f"tolerance, so this segment's movement is fully accounted for."
                )
            else:
                candidate.reason = (
                    f"Predicted {predicted:.6g} if the accused explains everything; observed "
                    f"{candidate.observed_value:.6g}. Residual {residual:+.6g} exceeds "
                    "tolerance, so this segment is not fully explained and may carry a "
                    "second cause."
                )

    def _overlap(
        self, metric: Metric, candidate: Candidate, accused: Candidate, history: HistoryCache
    ) -> float | None:
        """Fraction of the candidate's denominator that the accused occupies."""
        if accused.segment.is_total:
            return 1.0
        if candidate.segment == accused.segment:
            return 1.0

        combined_dims = set(candidate.segment.dimensions) | set(accused.segment.dimensions)
        if len(combined_dims) > 2:
            return None

        merged = dict(candidate.segment.keys)
        for k, v in accused.segment.keys:
            if merged.get(k, v) != v:
                # The two segments fix the same dimension to different values, so they cannot
                # overlap at all.
                return 0.0
            merged[k] = v
        overlap_segment = Segment(tuple(sorted(merged.items())))

        cells = history.combo(overlap_segment.combo)
        weeks = cells.get(overlap_segment)
        if weeks is None:
            return 0.0

        numerator = weeks[0].denominator(metric) if metric.is_ratio else weeks[0].requests
        denominator = (
            candidate.observed.denominator(metric)
            if metric.is_ratio
            else candidate.observed.requests
        )
        if not denominator:
            return None
        return min(1.0, (numerator or 0.0) / denominator)

    def _trace(
        self,
        metric: Metric,
        finding: Finding,
        parent_dev: float,
        candidates: list[Candidate],
        accused: Candidate | None,
        mode: str,
    ) -> None:
        assert self.tracer is not None
        with self.tracer.span(f"localize:{metric.name}", kind="localizer") as span:
            span.what(
                f"Tested {len(candidates)} candidate segments by removing each one from the "
                "population and recomputing the parent metric."
            )
            span.why(
                "Segments correlated with the cause move too, and often by a large and "
                "significant amount. Ranking cannot tell a cause from a passenger; a "
                "counterfactual can."
            )
            if accused is None:
                span.result(
                    "No candidate passed every applicable test. Reporting no verdict is the "
                    "correct outcome rather than naming the best of a bad set."
                )
            else:
                span.result(
                    f"{accused.segment.label()} accounts for {accused.sufficiency:.0%} of a "
                    f"{parent_dev:+.6g} movement and survived the breadth tests."
                )
            span.set("verdict.mode", mode)
            span.set("verdict.candidates", len(candidates))
            span.set("verdict.detector", finding.detector)
