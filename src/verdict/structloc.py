"""Naming a segment when the history cannot be trusted.

The counterfactual localizer asks, of every candidate, whether removing it returns the parent to
*expectation* -- and expectation comes from aligned weeks of history. When the baseline audit
rejects that history, because the population it describes is not the population in front of us,
every one of those tests is asking about a world that no longer exists. The pipeline is right to
refuse to name anyone on that evidence.

That refusal is honest and useless. The structural detector still works, because it compares a
cell against its own row and column inside a single window, but what it reports is a two-way
residual: ``country=IN x os_version=iOS 17.5 sits below what its row and column imply``. What an
operator needs to hear is ``iOS 17.5 is broken``.

Worse, the structural detector cannot say that on its own, by construction. Median polish sweeps
row and column medians out of the table, so a dimension value that is uniformly bad is absorbed
into its column effect and leaves residuals near zero in every grid that contains it. The cells
that do light up are in *other* grids, where the damage leaks in unevenly depending on how much
of the broken traffic each cell happens to carry. The detector is pointing at the shadow.

This module asks the counterfactual question with the baseline replaced by siblings. Instead of
"what did this segment do last Tuesday", it asks "what are the other values of this dimension
doing right now". The median across a dimension's levels is a within-window expectation: it
needs no history, so it survives the audit failure, and being a median it is robust to one level
having collapsed -- which is exactly the case being diagnosed.

The counterfactual is then identical in form to the historical one, and deliberately reuses the
same :func:`~verdict.localize.sufficiency_check` rather than reimplementing it. Remove the
candidate; see whether what remains reads like the sibling norm. The only thing that changes is
where the expected counters came from.

Two limits worth stating plainly, because both are structural rather than incidental.

**Only rates have siblings.** A sibling norm for a count is meaningless: segments differ in size
for entirely legitimate reasons, and "iOS 17.5 should have served as many requests as the median
os_version" is not a claim about health. Count metrics therefore get no verdict here at all,
rather than a confident one built on a comparison that does not hold.

**A uniform regression is invisible.** If every level of a dimension moved together, the median
moved with them and nothing stands out. That is the precise mirror of the temporal detector's
blind spot, and the reason the two are kept as separate detectors rather than merged: history
sees population-wide moves and cannot see mix, siblings see mix and cannot see population-wide
moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import DetectionConfig, LocalizationConfig
from .detect import Finding
from .localize import Candidate, Check, Localization, sufficiency_check
from .metrics import Metric, MetricRegistry
from .query import Counters, RollupReader, Segment, Window
from .schema import TOTAL_COMBO
from .stats import median, two_proportion_test
from .structural import cell_floor
from .trace import Tracer

log = logging.getLogger(__name__)

#: The ad funnel, in order. Scaling one stage means scaling everything it feeds, so that a
#: counterfactual "what if this cell had filled normally" leaves render rate and CTR where they
#: actually were instead of silently reshaping the whole funnel.
FUNNEL = ("requests", "fills", "impressions", "clicks")

#: Levels a dimension needs before its median says anything. With two, the median sits midway
#: between the healthy one and the broken one and neither looks unusual.
MIN_LEVELS = 3

#: How many candidates get the expensive two-way checks, in sufficiency order. Sufficiency is
#: free once the one-way slices are read; minimality and maximality each need the candidate split
#: by every other dimension, so they are spent only where they can change the answer.
MAX_EXAMINED = 5


def at_rate(counters: Counters, metric: Metric, target: float) -> Counters | None:
    """The counters this cell would show if its metric read ``target``.

    Scales the metric's numerator to hit the target and carries everything downstream along by
    the same factor, so the rest of the funnel keeps the shape it actually had. A cell that
    should have filled twice as often would also have served roughly twice the impressions;
    leaving impressions where they were would hand the counterfactual a render rate above one,
    ``Counters.coherent`` would reject it, and the sufficiency test that depends on subtracting
    these would quietly return "unknown" instead of an answer.

    Returns None rather than an approximation when the scaling cannot be expressed -- a cell with
    no traffic, or a target that would push a stage past the one feeding it.
    """
    current = counters.value(metric)
    if current is None or current <= 0 or target <= 0:
        return None

    scale = target / current
    field = metric.numerator_field
    if field == "revenue":
        touched = {"revenue"}
    else:
        idx = FUNNEL.index(field)
        touched = set(FUNNEL[idx:])
        # Revenue is earned per impression, so it follows the funnel down to impressions and
        # stops. A change in clicks does not imply a change in revenue.
        if idx <= FUNNEL.index("impressions"):
            touched.add("revenue")

    scaled = Counters(
        requests=_scaled_int(counters.requests, scale, "requests" in touched),
        fills=_scaled_int(counters.fills, scale, "fills" in touched),
        impressions=_scaled_int(counters.impressions, scale, "impressions" in touched),
        clicks=_scaled_int(counters.clicks, scale, "clicks" in touched),
        revenue=counters.revenue * scale if "revenue" in touched else counters.revenue,
    )
    return scaled if scaled.coherent else None


def _scaled_int(value: int, scale: float, apply: bool) -> int:
    return int(round(value * scale)) if apply else value


@dataclass
class Bucket:
    """One dimension's levels, and the rate a typical level is holding in this window."""

    dimension: str
    cells: dict[Segment, Counters]
    norm: float
    floor: float

    @property
    def levels(self) -> int:
        return len(self.cells)


def sibling_norm(cells: dict[Segment, Counters], metric: Metric, floor: float) -> float | None:
    """The rate a typical level of this dimension is running at.

    An unweighted median across levels, deliberately, on both counts.

    Unweighted, because weighting by traffic lets a large broken segment *become* the norm: with
    55% of requests sitting at a collapsed rate, the traffic-weighted median is the collapsed
    rate and the incident defines the baseline it would be measured against. The unweighted
    median survives any single level however large, which is the whole point of using it.

    A median rather than a mean, because the mean is dragged toward the outlier being looked for
    and partly erases the gap that identifies it -- the same reason
    :func:`~verdict.structural.median_polish` sweeps medians.

    Levels below ``floor`` are left out entirely. A segment with two hundred requests has a rate
    that is mostly noise, and admitting a handful of those pulls the median around far more than
    the incident does.
    """
    rates = []
    for counters in cells.values():
        den = counters.denominator(metric) or 0.0
        if den < floor:
            continue
        value = counters.value(metric)
        if value is not None and value > 0:
            rates.append(value)
    if len(rates) < MIN_LEVELS:
        return None
    return median(rates)


def sibling_minimality(
    metric: Metric,
    candidate: Candidate,
    norm: float,
    splits: list[dict[Segment, Counters]],
    threshold: float,
) -> Check:
    """Is the candidate carrying a smaller segment that is the real answer?

    The historical test removes the candidate's guiltiest child and asks whether what remains
    returns to its own past. With no past to return to, the question becomes whether it returns
    to the level its siblings are holding right now.

    Taken as the strictest result over every way of splitting the candidate, because this is an
    existence claim: if *any* single child can be removed and take the deviation with it, then
    the candidate is not the smallest true description and reporting it sends an operator to look
    at far more traffic than is actually broken.
    """
    deviation = (candidate.observed_value or 0.0) - norm
    if abs(deviation) < 1e-12:
        return Check(
            "minimality",
            "unknown",
            None,
            "This candidate sits at the sibling norm, so there is no deviation for a child to "
            "be carrying.",
        )
    if not splits:
        return Check(
            "minimality",
            "unknown",
            None,
            "No second dimension splits this candidate, so there is no smaller segment inside "
            "it to test.",
        )

    worst_score = None
    worst_child = ""
    for children in splits:
        guiltiest, remainder = _peel(metric, candidate.observed, children, norm)
        if guiltiest is None or remainder is None:
            continue
        residual = remainder - norm
        score = min(1.0, abs(residual) / abs(deviation))
        if worst_score is None or score < worst_score:
            worst_score = score
            worst_child = guiltiest

    if worst_score is None:
        return Check(
            "minimality",
            "unknown",
            None,
            "Every split of this candidate left a remainder too small to read a rate from.",
        )

    detail = (
        f"{candidate.segment.label()} sits {deviation:+.6g} from the sibling norm of "
        f"{norm:.6g}. With its worst child ({worst_child}) removed, {worst_score:.0%} of that "
        "gap is still there, so the child is not the whole story."
        if worst_score >= threshold
        else (
            f"{candidate.segment.label()} sits {deviation:+.6g} from the sibling norm of "
            f"{norm:.6g}, but removing {worst_child} leaves only {worst_score:.0%} of it. The "
            "smaller segment is the real answer."
        )
    )
    return Check("minimality", "pass" if worst_score >= threshold else "fail", worst_score, detail)


def _peel(
    metric: Metric,
    parent: Counters,
    children: dict[Segment, Counters],
    norm: float,
) -> tuple[str | None, float | None]:
    """Remove whichever child sits furthest from the norm, and report what is left."""
    guiltiest: Segment | None = None
    worst = 0.0
    for segment, counters in children.items():
        value = counters.value(metric)
        if value is None:
            continue
        gap = abs(value - norm) * (counters.denominator(metric) or 0.0)
        if gap > worst:
            worst = gap
            guiltiest = segment
    if guiltiest is None:
        return None, None
    remainder = parent - children[guiltiest]
    return guiltiest.label(), remainder.value(metric)


def sibling_maximality(
    metric: Metric,
    candidate: Candidate,
    norm: float,
    splits: list[dict[Segment, Counters]],
    threshold: float,
) -> Check:
    """Is the candidate a coherent whole, or is the damage concentrated in part of it?

    Measures the share of the candidate's traffic sitting in children that moved the same way it
    did. A genuine dimension-wide fault shows up in nearly all of them; a fault that is really
    about one country wearing an os_version as a disguise shows up in almost none.

    Summarised across splits by the median rather than the minimum. Unlike minimality this is a
    general claim about the candidate's shape, not an existence claim, and with seven partner
    dimensions the minimum is essentially the noisiest split every time.
    """
    deviation = (candidate.observed_value or 0.0) - norm
    if abs(deviation) < 1e-12:
        return Check(
            "maximality",
            "unknown",
            None,
            "This candidate sits at the sibling norm, so there is no movement for its children "
            "to share.",
        )
    if not splits:
        return Check(
            "maximality",
            "unknown",
            None,
            "No second dimension splits this candidate, so its breadth cannot be measured.",
        )

    shares = []
    for children in splits:
        agreed = 0.0
        total = 0.0
        for counters in children.values():
            den = counters.denominator(metric) or 0.0
            value = counters.value(metric)
            if value is None or den <= 0:
                continue
            total += den
            # Half the parent's own gap, in the same direction. A child barely off the norm is
            # not evidence of a shared fault, and one that moved the other way is evidence
            # against.
            if (value - norm) * deviation > 0 and abs(value - norm) >= abs(deviation) * 0.5:
                agreed += den
        if total > 0:
            shares.append(agreed / total)

    if not shares:
        return Check(
            "maximality", "unknown", None, "No child of this candidate carried readable traffic."
        )

    score = median(shares)
    detail = (
        f"{score:.0%} of {candidate.segment.label()}'s traffic sits in sub-segments that moved "
        f"with it, across {len(shares)} way(s) of splitting it. "
        + (
            "The fault is spread across the segment rather than hiding in one corner of it."
            if score >= threshold
            else "Most of the segment is behaving, so the label is broader than the fault."
        )
    )
    return Check("maximality", "pass" if score >= threshold else "fail", score, detail)


class SiblingLocalizer:
    """Localization against siblings in the same window, for when history is not usable."""

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
        self._slices: dict[str, dict[Segment, Counters]] = {}

    def localize(self, finding: Finding, *, direction: str | None = None) -> Localization:
        metric = self.registry.metric(finding.metric)
        window = finding.window
        parent = Segment.total()

        blank = Localization(
            metric=metric.name,
            window=window,
            parent=parent,
            parent_observed=None,
            parent_expected=None,
            parent_deviation=0.0,
            accused=None,
            mode="no_sibling_norm",
        )

        if not metric.is_ratio:
            return replace_note(
                blank,
                f"{metric.label} is a count, and counts have no sibling norm: segments differ in "
                "size for legitimate reasons, so 'this segment should be the size of the median "
                "segment' is not a claim about health. With the baseline rejected there is no "
                "honest way to localize this metric.",
            )

        parent_obs = self._slice(TOTAL_COMBO, window).get(parent)
        if parent_obs is None:
            return replace_note(blank, "No total-level rollup rows exist for this window.")

        buckets = self._buckets(metric, window)
        if not buckets:
            return replace_note(
                blank,
                "No dimension had enough comparable levels in this window to form a sibling "
                f"norm; each needs at least {MIN_LEVELS} above its detection floor.",
            )

        candidates, norms = self._build_candidates(metric, parent_obs, buckets)
        if not candidates:
            return replace_note(blank, "No level of any dimension carried readable traffic.")

        floor = metric.effect_threshold(self.detection.min_relative_effect)
        ranked = self._rank(candidates, direction, floor)
        for candidate in ranked[:MAX_EXAMINED]:
            norm = norms[candidate.segment]
            splits = self._splits(metric, candidate.segment, window)
            candidate.checks["minimality"] = sibling_minimality(
                metric, candidate, norm, splits, self.cfg.minimality_threshold
            )
            candidate.checks["maximality"] = sibling_maximality(
                metric, candidate, norm, splits, self.cfg.maximality_threshold
            )

        accused = self._accuse(ranked[:MAX_EXAMINED])
        parent_value = parent_obs.value(metric)
        parent_expected = norms[accused.segment] if accused else None
        confirmatory = self._confirm(metric, accused, buckets, window) if accused else None

        if self.tracer is not None:
            self._trace(metric, buckets, candidates, accused, parent_value)

        return Localization(
            metric=metric.name,
            window=window,
            parent=parent,
            parent_observed=parent_value,
            parent_expected=parent_expected,
            parent_deviation=(parent_value or 0.0) - (parent_expected or parent_value or 0.0),
            accused=accused,
            candidates=candidates,
            mode="siblings",
            note=(
                "Judged against siblings in the same window rather than against history, because "
                "the baseline audit rejected the history. Every number here is measured inside "
                f"{window.label()}."
            ),
            accused_finding=confirmatory,
        )

    def _slice(self, combo: str, window: Window) -> dict[Segment, Counters]:
        if combo not in self._slices:
            self._slices[combo] = self.reader.slice(combo, window)
        return self._slices[combo]

    def _buckets(self, metric: Metric, window: Window) -> list[Bucket]:
        out = []
        for dimension in self.registry.valid_dimensions(metric):
            cells = self._slice(dimension, window)
            if len(cells) < MIN_LEVELS:
                continue
            floor = cell_floor(metric, cells, self.detection)
            norm = sibling_norm(cells, metric, floor)
            if norm is None:
                continue
            out.append(Bucket(dimension, cells, norm, floor))
        return out

    def _build_candidates(
        self, metric: Metric, parent_obs: Counters, buckets: list[Bucket]
    ) -> tuple[list[Candidate], dict[Segment, float]]:
        candidates: list[Candidate] = []
        norms: dict[Segment, float] = {}

        for bucket in buckets:
            parent_exp = at_rate(parent_obs, metric, bucket.norm)
            if parent_exp is None:
                continue
            for segment, counters in bucket.cells.items():
                den = counters.denominator(metric) or 0.0
                if den < bucket.floor:
                    continue
                observed_value = counters.value(metric)
                expected = at_rate(counters, metric, bucket.norm)
                if observed_value is None or expected is None:
                    continue
                candidate = Candidate(
                    segment=segment,
                    observed=counters,
                    expected=expected,
                    observed_value=observed_value,
                    expected_value=bucket.norm,
                )
                candidate.checks["sufficiency"] = sufficiency_check(
                    metric, parent_obs, parent_exp, candidate, self.cfg.sufficiency_threshold
                )
                candidates.append(candidate)
                norms[segment] = bucket.norm

        return candidates, norms

    def _rank(self, candidates: list[Candidate], direction: str | None, floor: float) -> list[Candidate]:
        """Order by how much of the parent's gap each one explains.

        Tie-broken by the candidate's own distance from its siblings, not by its traffic. Volume
        is the wrong tie-break here: when two segments both explain the whole move it hands the
        verdict to the bigger one, which under a pure traffic shift is the healthy segment that
        merely shrank.
        """
        viable = []
        for candidate in candidates:
            if abs(candidate.relative_effect) < floor:
                candidate.status = "too_small"
                candidate.reason = (
                    f"Sits {candidate.relative_effect:+.1%} from its siblings, under the "
                    f"{floor:.1%} this metric needs to be worth reporting."
                )
                continue
            if direction == "drop" and candidate.deviation >= 0:
                candidate.status = "wrong_direction"
                candidate.reason = "Moved the opposite way to the incident."
                continue
            if direction == "rise" and candidate.deviation <= 0:
                candidate.status = "wrong_direction"
                candidate.reason = "Moved the opposite way to the incident."
                continue
            viable.append(candidate)

        viable.sort(key=lambda c: (-round(c.sufficiency, 6), -abs(c.relative_effect)))
        return viable

    def _accuse(self, ranked: list[Candidate]) -> Candidate | None:
        for candidate in ranked:
            if candidate.sufficiency < self.cfg.sufficiency_threshold:
                candidate.status = "partial"
                candidate.reason = (
                    f"Explains only {candidate.sufficiency:.0%} of the gap between the total and "
                    "the sibling norm."
                )
                continue
            if candidate.check_state("minimality") == "fail":
                candidate.status = "too_broad"
                candidate.reason = candidate.checks["minimality"].detail
                continue
            if candidate.check_state("maximality") == "fail":
                candidate.status = "too_broad"
                candidate.reason = candidate.checks["maximality"].detail
                continue
            candidate.status = "accused"
            return candidate
        return None

    def _splits(
        self, metric: Metric, segment: Segment, window: Window
    ) -> list[dict[Segment, Counters]]:
        """The candidate broken down by every other dimension in turn."""
        keys = segment.as_dict()
        if len(keys) != 1:
            return []
        dimension = next(iter(keys))
        out = []
        for other in self.registry.valid_dimensions(metric):
            if other == dimension:
                continue
            combo = "|".join(sorted((dimension, other)))
            children = {
                child: counters
                for child, counters in self._slice(combo, window).items()
                if _contains(child, keys)
            }
            if len(children) >= 2:
                out.append(children)
        return out

    def _confirm(
        self, metric: Metric, accused: Candidate, buckets: list[Bucket], window: Window
    ) -> Finding | None:
        """Test the accused cell against its siblings pooled, inside this window.

        The detector's own p-value came from a median-polish residual on some other grid, which
        is not evidence about this claim. This is: a two-proportion test between the accused
        segment and the rest of its own dimension, with both arms measured in the same window and
        neither borrowed from a baseline this run has rejected.
        """
        if not metric.is_proportion:
            return None
        keys = accused.segment.as_dict()
        if len(keys) != 1:
            return None
        dimension = next(iter(keys))
        bucket = next((b for b in buckets if b.dimension == dimension), None)
        if bucket is None:
            return None

        rest = Counters()
        for segment, counters in bucket.cells.items():
            if segment != accused.segment:
                rest = rest + counters

        n_obs = accused.observed.denominator(metric) or 0.0
        n_rest = rest.denominator(metric) or 0.0
        if n_obs <= 0 or n_rest <= 0:
            return None

        test = two_proportion_test(
            accused.observed.numerator(metric), n_obs, rest.numerator(metric), n_rest
        )
        return Finding(
            metric=metric.name,
            segment=accused.segment,
            window=window,
            detector="siblings",
            test=test,
            observed_counters=accused.observed,
            baseline_counters=rest,
            phi=1.0,
            effect_threshold=metric.effect_threshold(self.detection.min_relative_effect),
            screening="sibling_pooled",
            notes={"dimension": dimension, "sibling_norm": bucket.norm, "levels": bucket.levels},
        )

    def _trace(
        self,
        metric: Metric,
        buckets: list[Bucket],
        candidates: list[Candidate],
        accused: Candidate | None,
        parent_value: float | None,
    ) -> None:
        if self.tracer is None:
            return
        with self.tracer.span(f"siblings:{metric.name}", kind="localizer") as span:
            span.what(
                f"Compared every level of {len(buckets)} dimension(s) against the median level of "
                f"its own dimension, inside {metric.label}'s own window, and removed each in turn "
                "to see which one brings the total back to that median."
            )
            span.why(
                "The baseline audit rejected this run's history, so every counterfactual against "
                "the past is asking about a population that no longer exists. Siblings in the "
                "same window need no history and are robust to one level having collapsed."
            )
            if accused is None:
                span.result(
                    f"{len(candidates)} candidate(s) considered, none sufficient on its own."
                )
                return
            span.result(
                f"{accused.segment.label()} reads {accused.observed_value:.6g} against a sibling "
                f"median of {accused.expected_value:.6g}, and removing it accounts for "
                f"{accused.sufficiency:.0%} of the gap between the total "
                f"({parent_value:.6g}) and that median."
                if parent_value is not None
                else f"{accused.segment.label()} accused."
            )


def _contains(child: Segment, keys: dict[str, str]) -> bool:
    child_keys = child.as_dict()
    return all(child_keys.get(k) == v for k, v in keys.items())


def replace_note(localization: Localization, note: str) -> Localization:
    localization.note = note
    return localization
