"""Confidence: one number, and the arithmetic that produced it.

An operator needs a single number to threshold on, and a single number is the easiest place in
the system to hide a claim nobody can check. So this produces one, but never produces it in a
form that cannot be read back line by line: five components, each carrying its own score, the
weight it actually contributed, and a sentence saying what it measured.

The hard part is not the weighting. It is what to do with a component that could not be
measured at all. Every localization check returns pass, fail, or **unknown**, and unknown
arrives from ordinary structural facts rather than from anything going wrong: a segment too
thin to split cannot be holdout-tested, a two-dimensional candidate has no three-dimensional
child to remove, a parent that did not move offers sufficiency nothing to explain.

Both obvious ways of handling that are wrong, and they are wrong in opposite directions.

Scoring an unknown as 0.0 punishes a segment for the system's own inability to test it. The
untestable segments are the small ones, so this quietly suppresses findings in exactly the
corner of the lattice an operator most needs told about, and the suppression is invisible
because the output still reads as a confident low score rather than as a gap.

Scoring an unknown as 1.0 rewards ignorance. The least testable candidate in the lattice would
come back the most confident one, and the thinner the evidence the better the verdict would
look.

So an unknown withdraws its own weight and the remaining weights are renormalised to sum to
1.0. The score lands wherever the evidence that was actually gathered puts it, neither dragged
down nor lifted up by the hole. That leaves a reporting obligation, because 0.8 from five
components and 0.8 from three are different claims: the count of what was scored, the effective
weights after renormalisation, and a caveat naming what went unmeasured all travel with the
number, and a score resting on too little of the weight is capped so it cannot reach into the
top of the range on the strength of what was never looked at.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

from .config import ConfidenceConfig
from .detect import Finding
from .localize import Candidate, Localization

#: Fixed evaluation order, so two runs over the same inputs emit components in the same
#: sequence and the case file diffs cleanly. Iterating ``cfg.weights`` instead would tie the
#: report layout to the order somebody happened to write the YAML in.
_ORDER = ("significance", "sufficiency", "minimality", "stability", "separation")

# Evidence is counted in decades of p-value rather than in p, because p is not linear in what
# it tells you: 0.05 and 0.01 make nearly the same claim, while 1e-3 and 1e-9 do not. Zero
# credit at p = 0.1, full credit at p = 1e-6. Those anchors are also what separates a finding
# that scraped past the false-discovery correction from one that cleared it by orders of
# magnitude -- at the family thresholds this system uses, the first sits near two decades and
# scores about 0.2, the second sits past six and scores 1.0.
_EVIDENCE_FLOOR_DECADES = 1.0
_EVIDENCE_FULL_DECADES = 6.0

# ``normal_sf`` deliberately returns 0.0 rather than losing precision past about z = 39, so a
# p-value of exactly zero means "smaller than a double can hold", not "impossible". Substituting
# a floor keeps the logarithm finite; nothing beyond this point changes any verdict.
_MIN_P = 1e-300

# Dispersion at which the statistical component is halved. The p-value has already been widened
# by phi, so this is not the same penalty twice: phi is itself estimated from a handful of
# weekly samples per cell, and a cell needing a tenfold variance inflation is one whose own
# history is unstable enough that the model, rather than the data, is carrying the result.
_PHI_HALF_LIFE = 10.0

# A finding the family-wide correction rejected cannot supply strong statistical evidence
# whatever its nominal p-value, because the p-value is precisely the quantity the correction
# has already discounted for the number of chances noise had.
_UNCORRECTED_CAP = 0.25

#: Why this component was capped, keyed by how the finding was screened. Three different
#: situations were previously reported with one sentence that was only true of the first.
_WHY_CAPPED = {
    "": (
        "The finding did not survive the false-discovery correction over its family, so this "
        "component is capped at {cap} whatever its nominal p-value."
    ),
    "benjamini_hochberg": (
        "The finding did not survive the false-discovery correction over its family, so this "
        "component is capped at {cap} whatever its nominal p-value."
    ),
    "structural_z": (
        "Screened against a fixed structural threshold rather than through the false-discovery "
        "correction, which runs over the temporal family only, so this component is capped at "
        "{cap} whatever its nominal p-value."
    ),
    "post_hoc": (
        "Re-tested after the localizer had already named this cell, so the p-value carries a "
        "selection effect no correction here accounts for, and this component is capped at "
        "{cap}. The localization gates, not this number, are the argument."
    ),
}

# Models ``stats`` returns when no test was performed. They carry p = 1.0 and z = 0.0, which
# would otherwise read as "tested and found unremarkable" rather than "never tested".
_UNTESTED_MODELS = frozenset({"log_ratio_insufficient", "log_ratio_degenerate"})

# Four aligned historical weeks with the most extreme one trimmed, which is what the shipped
# detection config produces. Fewer means the expectation both halves of a holdout are compared
# against is itself thin.
_FULL_BASELINE_WEEKS = 3

# The most a thin baseline may cost the stability component. An effect that reproduces on a
# fresh half of the window is still evidence when the baseline is one week; it is weaker
# evidence, not absent evidence, and zeroing it would discard a genuine reproduction.
_THIN_BASELINE_FLOOR = 0.5

# Candidates that competed for the accusation on the same footing and lost. ``too_broad`` and
# ``too_narrow`` are excluded because a counterfactual removed them for cause rather than on a
# margin, and treating them as near-misses would report low separation for the one shape the
# breadth tests handle best: a one-way parent that is only guilty through its child. ``cleared``
# is excluded because the exoneration ledger already predicted its movement and was right, and
# penalising the accusation for a successful clearance inverts the evidence.
_RIVAL_STATUSES = frozenset({"considered", "partial"})

# Share of the configured weight that has to be scored before the score may use the full range.
# With the shipped weights this is exactly what remains when the single heaviest component is
# unknown: four components still bound the answer from four directions, so one hole costs
# nothing. Two holes leave a picture too partial to earn the top of the range, and the ceiling
# falls in proportion to how much of the weight went unmeasured.
_MIN_FULL_COVERAGE = 0.70

# Below this, most of the distinct ways a verdict can be wrong -- noise, wrong scope, wrong
# window, wrong candidate -- were never examined, and a high score over the remainder says
# nothing about the ones that were skipped. Such a case is still published as a score with its
# caveat; it is just not allowed to claim it is publishable.
_MIN_PUBLISHABLE_COMPONENTS = 3

# Reported to six decimals so the number written into a case file, compared against a threshold
# and shown to an operator is the same number in all three places, rather than three
# presentations of a float whose last bits depend on summation order.
_VALUE_DP = 6

_NO_ACCUSED = (
    "No candidate was accused, so there is nothing for this component to measure. Absence of a "
    "verdict is not a weak verdict."
)


@dataclass(frozen=True)
class Component:
    """One axis of the score, after weights are known.

    ``score`` is 0.0 for an unknown component and ``weight`` is 0.0 alongside it, so an unknown
    cannot enter the weighted sum by arithmetic accident. Readers must branch on ``state``
    rather than on the score: a scored 0.0 and an unscored 0.0 are the two claims this module
    exists to keep apart.
    """

    name: str
    score: float
    weight: float
    state: str
    detail: str

    @property
    def scored(self) -> bool:
        return self.state == "scored"


@dataclass(frozen=True)
class Confidence:
    """A score and everything needed to recompute it by hand."""

    value: float
    components: list[Component]
    components_scored: int
    components_total: int
    publishable: bool
    caveat: str

    def explain(self) -> str:
        """The breakdown as it belongs in a case file.

        Written for someone who was not watching the investigation and has to decide whether to
        act on it. Every component appears whether or not it was scored, because a component
        omitted from the list would be indistinguishable from one that scored badly, and the
        effective weight appears next to the score so the renormalisation can be checked with a
        calculator rather than taken on trust.
        """
        verdict = "publishable" if self.publishable else "not publishable"
        lines = [
            f"Confidence {self.value:.4f} of 1.0000 -- {verdict}. "
            f"{self.components_scored} of {self.components_total} components scored.",
            "",
        ]
        for component in self.components:
            shown = f"{component.score:.4f}" if component.scored else "not scored"
            lines.append(f"  {component.name:<13}{shown:>10}   weight {component.weight:.4f}")
            lines.extend(
                textwrap.wrap(
                    component.detail, width=98, initial_indent=" " * 6, subsequent_indent=" " * 6
                )
            )
        if self.caveat:
            lines.append("")
            lines.extend(textwrap.wrap(f"Caveat: {self.caveat}", width=98))
        return "\n".join(lines)


@dataclass(frozen=True)
class _Reading:
    """One component before the weights are known.

    ``score is None`` is the unknown state, carried as an absence rather than as a sentinel
    number so that no arithmetic path can consume it by mistake.
    """

    score: float | None
    detail: str

    @property
    def known(self) -> bool:
        return self.score is not None


def _clamp(value: float) -> float:
    """Force a component onto [0, 1], treating a non-finite score as no evidence.

    A NaN arriving from a degenerate division would otherwise poison the weighted sum and turn
    the whole verdict into a quantity that compares false against every threshold, including
    the one that decides whether to publish.
    """
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _significance(finding: Finding) -> _Reading:
    """How much statistical evidence the detector actually had.

    Measured in decades of p-value, then discounted twice for things a p-value alone does not
    say: how overdispersed the cell was, and whether the finding survived correction over the
    family it was drawn from. Both discounts push the same way, because both describe evidence
    that is weaker than its nominal p-value suggests.
    """
    test = finding.test
    if test.model in _UNTESTED_MODELS:
        return _Reading(
            None,
            f"No significance test was performed ({test.model}): the segment's own history was "
            "too short or too flat to test against. That is the absence of a test, not a "
            "p-value of one.",
        )

    p = test.p_value
    if not math.isfinite(p) or p > 1.0:
        p = 1.0
    decades = -math.log10(max(p, _MIN_P))
    span = _EVIDENCE_FULL_DECADES - _EVIDENCE_FLOOR_DECADES
    ramp = _clamp((decades - _EVIDENCE_FLOOR_DECADES) / span)

    phi = finding.phi if math.isfinite(finding.phi) else 1.0
    phi_factor = 1.0 / (1.0 + math.log10(max(phi, 1.0)) / math.log10(_PHI_HALF_LIFE))
    value = ramp * phi_factor

    detail = (
        f"p = {p:.3g} is {decades:.1f} decades of evidence, {ramp:.0%} of the way from the "
        f"floor at p = {10 ** -_EVIDENCE_FLOOR_DECADES:g} to full credit at "
        f"p = {10 ** -_EVIDENCE_FULL_DECADES:g}."
    )
    if phi_factor < 1.0:
        detail += (
            f" Dispersion measured at {phi:.2f}x keeps {phi_factor:.0%} of that, because the "
            "variance model the p-value rests on is itself estimated from a few weekly samples "
            "per cell."
        )
    if not finding.survives_correction:
        # Capped either way, but the reason differs and stating the wrong one is a false claim
        # about how the finding was arrived at. Only a finding that actually entered the family
        # and lost can be said to have failed the correction; the others were never offered to
        # it, for reasons that are themselves part of the argument.
        value = min(value, _UNCORRECTED_CAP)
        detail += f" {_WHY_CAPPED.get(finding.screening, _WHY_CAPPED[''])}".format(
            cap=f"{_UNCORRECTED_CAP:.2f}"
        )
    return _Reading(_clamp(value), detail)


def _sufficiency(localization: Localization, accused: Candidate) -> _Reading:
    """How much of the parent's movement disappears when this segment is removed.

    Passed through from the localization check rather than rescaled, because the check's score
    is already the fraction of the deviation accounted for -- the quantity an operator would
    recompute by hand -- and rescaling it against the pass threshold would make a marginal pass
    read like a comfortable one.
    """
    if localization.mode == "structural_only":
        return _Reading(
            None,
            "The parent metric did not move, so there is no parent deviation for this segment "
            "to account for and the counterfactual was never run. Scoring that as a failure "
            "would penalise the candidate for the shape of the incident.",
        )

    check = accused.checks.get("sufficiency")
    if check is None:
        return _Reading(None, "The sufficiency counterfactual was not run on this candidate.")
    if check.state == "unknown" or check.score is None:
        return _Reading(None, check.detail)
    return _Reading(_clamp(check.score), check.detail)


def _minimality(accused: Candidate) -> _Reading:
    """Whether this is the right level to name, or a child is carrying the whole deviation.

    The maximality check bounds the same question from the other side, and is folded in only
    when it failed. The localizer already refuses to accuse a candidate its siblings moved with,
    so this branch should be unreachable through the normal path -- but a caller scoring a
    candidate directly would otherwise be told the level is right on the strength of one bound
    while the other one was rejecting it.
    """
    check = accused.checks.get("minimality")
    if check is None:
        return _Reading(None, "The minimality counterfactual was not run on this candidate.")
    if check.state == "unknown" or check.score is None:
        return _Reading(None, check.detail)

    value = _clamp(check.score)
    detail = check.detail
    breadth = accused.checks.get("maximality")
    if breadth is not None and breadth.state == "fail" and breadth.score is not None:
        capped = min(value, _clamp(breadth.score))
        if capped < value:
            detail += (
                f" Held down to {capped:.2f} by the maximality check, which found the movement "
                "wider than this candidate. The two checks bound the level from opposite sides "
                "and the weaker bound is the one worth reporting."
            )
            value = capped
    return _Reading(value, detail)


def _baseline_factor(finding: Finding) -> tuple[float, str]:
    """How far the quality of the baseline discounts a reproduction on the held-out half.

    A holdout compares each half against an expectation built from the same weekly history, so a
    thin baseline makes both halves noisy at once. That degrades the reproduction rather than
    constituting separate evidence, which is why it multiplies the holdout score instead of
    scoring as its own component.

    A finding that recorded no weekly baseline at all -- the structural detector fits a median
    polish across the grid instead -- is not discounted. Charging it for a field its detector
    never populates would score an absence as a defect.
    """
    if finding.weeks_seen <= 0:
        return 1.0, ""
    kept = max(0, min(finding.weeks_kept, _FULL_BASELINE_WEEKS))
    factor = _THIN_BASELINE_FLOOR + (1.0 - _THIN_BASELINE_FLOOR) * kept / _FULL_BASELINE_WEEKS
    if factor >= 1.0:
        return 1.0, ""
    note = (
        f" Discounted to {factor:.0%} because the baseline kept {finding.weeks_kept} of "
        f"{finding.weeks_seen} historical weeks, so the expectation each half was compared "
        "against is itself thin."
    )
    return factor, note


def _stability(accused: Candidate, finding: Finding) -> _Reading:
    """Whether the effect reproduced on the half of the window it was not selected on.

    When the holdout could not be run the component is unknown, and deliberately not scored from
    the baseline quality alone. Weeks of history say something about the expectation; they say
    nothing about whether this effect reproduces, and reporting a stability number derived only
    from them would claim a check that never happened.
    """
    check = accused.checks.get("holdout")
    if check is None:
        return _Reading(
            None,
            "The window was never split, either because holdout testing is switched off or "
            "because this candidate was not the one carried forward to it.",
        )
    if check.state == "unknown" or check.score is None:
        return _Reading(None, check.detail)

    factor, note = _baseline_factor(finding)
    return _Reading(_clamp(check.score) * factor, check.detail + note)


def _sufficiency_of(candidate: Candidate) -> float | None:
    """A candidate's sufficiency, or None when it was never scored.

    ``Candidate.sufficiency`` reports 0.0 for both, which is right for ranking and wrong here:
    an unscored rival would look like a rival that explains nothing, and the accused would take
    credit for a separation from a candidate nobody measured.
    """
    check = candidate.checks.get("sufficiency")
    if check is None or check.score is None:
        return None
    return check.score


def _deviation_of(candidate: Candidate) -> float | None:
    if not candidate.expected_value:
        return None
    return abs(candidate.relative_effect)


def _separation(localization: Localization, accused: Candidate) -> _Reading:
    """How far clear of the runner-up the accusation stands.

    Every other check grades the accused in isolation, and all of them can pass while a second
    candidate passes just as well. That is the case where the verdict is most likely to name the
    wrong segment, and nothing else in the score can see it.

    The margin is relative rather than absolute, because "nearly as good" is a ratio: a rival at
    0.80 against an accused at 0.90 is a much closer call than one at 0.10 against 0.20, though
    both are 0.10 apart.
    """
    # Compared on the segment rather than on object identity, because the accused reaching this
    # function through a serialised case file is a different object naming the same cell, and a
    # candidate cannot be a rival to itself.
    others = [c for c in localization.candidates if c.segment != accused.segment]
    if not others:
        return _Reading(
            None,
            "No other candidate was enumerated, so there was no field to stand apart from. "
            "Recorded as untested rather than as a clear win over nobody.",
        )

    rivals = [c for c in others if c.status in _RIVAL_STATUSES]
    if not rivals:
        return _Reading(
            1.0,
            f"All {len(others)} other candidates were either excluded by a breadth check or "
            "exonerated by the ledger, so nothing was left competing for the accusation.",
        )

    if localization.mode == "explain_away" and _sufficiency_of(accused) is not None:
        measure, basis = _sufficiency_of, "share of the parent movement explained"
    else:
        measure, basis = _deviation_of, "relative deviation"

    top = measure(accused)
    if top is None or top <= 0.0:
        return _Reading(
            None,
            f"The accused segment has no measurable {basis} to compare rivals against, so the "
            "margin between it and the field is undefined.",
        )

    measurable = [(q, c) for q, c in ((measure(c), c) for c in rivals) if q is not None]
    if not measurable:
        return _Reading(
            None,
            f"{len(rivals)} candidate(s) were still in contention but none could be placed on "
            f"the same {basis} scale, so the margin could not be measured.",
        )

    # Ties broken on the segment label so that two rivals reaching the same number always name
    # the same one in the case file. Iteration order of the candidate list is stable, but a
    # reader comparing two runs should not have to know that.
    best_score, best = min(measurable, key=lambda pair: (-pair[0], pair[1].segment.label()))
    margin = _clamp((top - best_score) / top)
    detail = (
        f"The closest rival, {best.segment.label()}, reaches {best_score:.3g} against the "
        f"accused segment's {top:.3g} on {basis} -- a relative margin of {margin:.0%} over "
        f"{len(measurable)} candidate(s) still in contention."
    )
    return _Reading(margin, detail)


def _coverage_ceiling(coverage: float) -> float:
    """The highest value a score resting on this much of the weight is allowed to reach.

    One-sided by construction. Renormalisation already stops an unknown from dragging the score
    down, and the ceiling stops it from lifting the score up: a candidate measured on a quarter
    of the weight cannot reach the top of the range on the strength of the three quarters nobody
    looked at. It never raises a score, so a weak case stays weak.
    """
    if coverage >= _MIN_FULL_COVERAGE:
        return 1.0
    return coverage / _MIN_FULL_COVERAGE


def _caveat(
    unknown_names: list[str], coverage: float, raw: float, ceiling: float, scored: int
) -> str:
    parts: list[str] = []
    if scored == 0:
        return (
            "Not one component could be evaluated, so this is the absence of a score rather "
            "than a low one. Nothing here says the accusation is wrong; it says nothing at all."
        )
    if unknown_names:
        parts.append(
            f"{len(unknown_names)} of {len(_ORDER)} components could not be evaluated "
            f"({', '.join(unknown_names)}), so their weight was removed and the rest "
            f"renormalised. This score rests on {coverage:.0%} of the evidence a complete case "
            "carries."
        )
    if ceiling < 1.0 and raw > ceiling:
        parts.append(
            f"Capped at {ceiling:.4f} from an uncapped {raw:.4f} because only {coverage:.0%} of "
            "the configured weight was scored."
        )
    if scored < _MIN_PUBLISHABLE_COMPONENTS:
        parts.append(
            f"Withheld from publication regardless of the value: fewer than "
            f"{_MIN_PUBLISHABLE_COMPONENTS} components were scored, so most of the ways this "
            "verdict could be wrong were never checked."
        )
    return " ".join(parts)


def _assemble(readings: dict[str, _Reading], cfg: ConfidenceConfig) -> Confidence:
    """Renormalise around whatever could be measured, then report what that cost."""
    configured = {name: float(cfg.weights.get(name, 0.0)) for name in _ORDER}
    total_weight = sum(configured.values())
    scored_weight = sum(configured[name] for name in _ORDER if readings[name].known)

    components: list[Component] = []
    for name in _ORDER:
        reading = readings[name]
        effective = configured[name] / scored_weight if reading.known and scored_weight else 0.0
        components.append(
            Component(
                name=name,
                score=reading.score if reading.score is not None else 0.0,
                weight=effective,
                state="scored" if reading.known else "unknown",
                detail=reading.detail,
            )
        )

    raw = sum(c.score * c.weight for c in components)
    coverage = scored_weight / total_weight if total_weight > 0 else 0.0
    ceiling = _coverage_ceiling(coverage)
    value = round(_clamp(min(raw, ceiling)), _VALUE_DP)

    scored = sum(1 for c in components if c.scored)
    unknown_names = [c.name for c in components if not c.scored]
    # Significance is not one vote among five. The other four all ask "is this the right
    # culprit", and every one of them can be answered convincingly about a movement that was
    # never shown to be a movement at all -- a segment can explain a parent's wobble, be at the
    # right level, and beat every rival, while the whole thing is noise. Significance is the
    # only component that asks whether there is anything to explain, so a verdict without it is
    # not a weaker verdict, it is an answer to a question nobody established.
    tested = any(c.name == "significance" and c.scored for c in components)
    publishable = (
        value >= cfg.publish_threshold and scored >= _MIN_PUBLISHABLE_COMPONENTS and tested
    )
    return Confidence(
        value=value,
        components=components,
        components_scored=scored,
        components_total=len(_ORDER),
        publishable=publishable,
        caveat=_caveat(unknown_names, coverage, raw, ceiling, scored),
    )


def score(localization: Localization, finding: Finding, cfg: ConfidenceConfig) -> Confidence:
    """Grade one accusation on five components and combine them into a number to threshold on.

    The five are not five measurements of one quantity. They are five separate ways for the
    verdict to be wrong -- it is noise, it does not explain the parent, it names the wrong level,
    it is an artefact of the window, it names the wrong one of several equally good candidates
    -- which is why a component that could not be measured is a hole in the argument rather than
    a low mark, and why the count of holes is published next to the number.
    """
    accused = localization.accused
    if accused is None:
        return _assemble({name: _Reading(None, _NO_ACCUSED) for name in _ORDER}, cfg)

    # Both components below read the detector's test, and a test describes exactly one cell. The
    # finding a case is entered on is whichever cell had the smallest p-value in the group, which
    # is frequently *not* the segment localization goes on to name. Scoring them anyway is the
    # worst failure this module can produce: an accusation certified by evidence gathered about
    # somebody else. It reads as the strongest possible case -- five components, no caveat -- and
    # the number that convinced you was measured on a different segment.
    #
    # Withheld rather than substituted. The absence of a test for the accused is a real hole in
    # the argument, and the coverage rule already prices holes.
    own = finding.segment == accused.segment
    borrowed = _Reading(
        None,
        f"No test was run on {accused.segment.label()} itself. The detector's evidence here "
        f"describes {finding.segment.label()}, a different cell, so it cannot speak to this "
        "accusation.",
    )

    return _assemble(
        {
            "significance": _significance(finding) if own else borrowed,
            "sufficiency": _sufficiency(localization, accused),
            "minimality": _minimality(accused),
            "stability": _stability(accused, finding) if own else borrowed,
            "separation": _separation(localization, accused),
        },
        cfg,
    )
