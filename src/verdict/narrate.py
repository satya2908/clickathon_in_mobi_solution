"""Turning a finished investigation into prose without letting the model near a number.

The property this module exists to protect: switch the model off and every figure in every
case file is byte-identical, and only the prose degrades to a template. That is not a claim
about prompt discipline, which is unenforceable. It is enforced twice. The model is handed
pre-computed claim tuples and never the underlying data, so it has nothing to compute from;
and whatever it writes is then re-read by `verify_numbers`, which discards the draft if a
single figure in it cannot be traced to a tuple. A discarded draft is replaced by the
template and the offending literals are recorded on the result, so a caught fabrication is
visible in the case file rather than silently absorbed.

Most of the verifier's difficulty is not arithmetic. Prose is full of digits that are not
figures: the year in a date, the minutes in a timestamp, the ordinal in "the 2nd half", and
above all the digits inside segment names -- Android 15, iOS 17.2, Galaxy S23. A verifier that
flags those rejects every correct narration this system will ever produce, and the fallback
fires so reliably that the model becomes decoration nobody notices is switched off. That is a
worse outcome than having no verifier, because it looks like one. See `_mask_non_figures`.

The template is not a degraded placeholder. It is what ships whenever the model is off,
unreachable, or caught inventing, so it has to be prose an on-call engineer can act on.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import LLMConfig
from .detect import Finding
from .llm import Completion, LLMClient
from .localize import Candidate, Localization

log = logging.getLogger(__name__)

# Units a claim may carry. The set is small on purpose: it exists so the verifier can refuse to
# satisfy a "35.2pp" in the prose with a value that is a percent, which is the single most
# common way a numerically faithful sentence still says something false.
Unit = str
PERCENT = "%"
POINTS = "pp"
PROBABILITY = "probability"
CURRENCY = "currency"
COUNT = "count"
PLAIN = ""

#: How a finding reached the published list. Spelled out rather than reported as a yes/no on
#: "survived the correction", because the two detectors are screened by different procedures
#: and a boolean forced the structural one to answer a question that was never asked of it.
_SCREENING_LABEL = {
    "benjamini_hochberg": "Benjamini-Hochberg over every cell tested in this run",
    "structural_z": "fixed structural threshold, outside the correction family",
    "post_hoc": "re-tested after the localizer named the cell, so outside the family",
}

# What a literal written with each suffix is allowed to resolve to. A bare literal may match
# anything, because prose legitimately writes a proportion as "0.93"; a suffixed one may not,
# because the suffix is an assertion about units that the claim can contradict.
_PERCENT_UNITS = frozenset({PERCENT, PROBABILITY})
_POINT_UNITS = frozenset({POINTS})
_CURRENCY_UNITS = frozenset({CURRENCY, PLAIN})
_PROPORTIONAL_UNITS = frozenset({PERCENT, POINTS, PROBABILITY})


@dataclass(frozen=True)
class Claim:
    """One pre-computed figure the model is permitted to phrase, and nothing else.

    ``detail`` carries provenance rather than commentary -- which test produced the figure and
    over what window -- because a claim a reader cannot trace back to a computation is exactly
    the thing this module exists to prevent, whether a model wrote it or not.
    """

    key: str
    label: str
    value: float | str
    unit: Unit = PLAIN
    detail: str = ""

    @property
    def numeric(self) -> float | None:
        """The value as a finite float, or None for a textual or unusable claim.

        Booleans are excluded explicitly. ``isinstance(True, int)`` is true in Python, so a
        state flag would otherwise enter the supported set as the number 1 and quietly excuse
        every stray "1" in the prose.
        """
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            return None
        out = float(self.value)
        return out if math.isfinite(out) else None

    @property
    def display(self) -> str:
        """The figure as the prose should render it.

        The model is given this string rather than the raw value so that it never has to
        multiply by a hundred, choose a precision, or decide whether something is a percent.
        Every conversion the model is forbidden from doing has already been done here.
        """
        return _display(self.value, self.unit)


@dataclass(frozen=True)
class EvidenceCheck:
    """One localization test as it appears in the evidence bundle."""

    name: str
    state: str
    score: float | None
    detail: str


@dataclass(frozen=True)
class ExonerationEntry:
    """One line of the exoneration ledger: what a cleared segment should have read, and did.

    Published so that clearing a candidate is falsifiable. "We looked and it was fine" is not
    checkable; "if the accusation holds this segment must read 0.644, and it reads 0.644" is.
    """

    segment: str
    predicted: float | None
    observed: float | None
    residual: float | None
    unit: Unit
    reason: str


@dataclass(frozen=True)
class RuledOut:
    """A candidate rejected by a breadth test rather than by prediction."""

    segment: str
    status: str
    reason: str


@dataclass(frozen=True)
class CoverageEntry:
    """Something that could not be tested, and the smallest move it could have resolved.

    Silence about a segment and evidence of its innocence are different claims. A reader who
    cannot tell them apart will eventually be confident about the wrong one.
    """

    segment: str
    reason: str
    resolvable_effect: float | None
    detail: str


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the narration layer is allowed to say, and the only thing the model sees.

    This is also the ``evidence.json`` deliverable, which is why `to_dict` is part of the
    contract rather than a debugging convenience: the figures in the prose and the figures a
    reviewer re-checks have to come from the same object, or the check proves nothing.
    """

    metric: str
    window_start: str
    window_end: str
    grain: str
    detector: str
    headline: str
    claims: list[Claim] = field(default_factory=list)
    checks: list[EvidenceCheck] = field(default_factory=list)
    cleared: list[ExonerationEntry] = field(default_factory=list)
    ruled_out: list[RuledOut] = field(default_factory=list)
    coverage: list[CoverageEntry] = field(default_factory=list)
    # Every proper noun the prose may legitimately contain. Carried on the bundle rather than
    # rebuilt by the verifier so that the allow-list is part of the audited artefact: a reader
    # can see exactly which strings were exempted from numeric checking.
    labels: tuple[str, ...] = ()
    mode: str = "explain_away"
    note: str = ""

    def figures(self) -> tuple[tuple[float, Unit], ...]:
        """Every figure the prose may state, paired with the unit it may state it in.

        Drawn from the claims, from the structured ledgers, and from the free text of every
        provenance string. The last of those is not laxity: those strings are rendered by the
        statistics code, they are printed in the case file verbatim, and the model is shown
        them. A model quoting "99% of the change is accounted for" out of a sufficiency detail
        is quoting a computed figure, and a verifier that rejected it would be rejecting the
        most useful sentence in the report.
        """
        out: list[tuple[float, Unit]] = []
        for claim in self.claims:
            value = claim.numeric
            if value is not None:
                out.append((value, claim.unit))
        for check in self.checks:
            if check.score is not None and math.isfinite(check.score):
                out.append((check.score, PROBABILITY))
        for entry in self.cleared:
            change_unit = _change_unit(entry.unit)
            for value, unit in (
                (entry.predicted, entry.unit),
                (entry.observed, entry.unit),
                (entry.residual, change_unit),
            ):
                if value is not None and math.isfinite(value):
                    out.append((value, unit))
        for gap in self.coverage:
            if gap.resolvable_effect is not None and math.isfinite(gap.resolvable_effect):
                out.append((gap.resolvable_effect, PERCENT))
        for text in self._provenance():
            out.extend(_figures_in_text(text, self.labels))
        return tuple(out)

    def numbers(self) -> set[float]:
        """Every numeric value the bundle carries, without units. Useful for coarse checks."""
        return {value for value, _ in self.figures()}

    def _provenance(self) -> list[str]:
        """Every deterministically generated string the model is shown."""
        texts = [self.headline, self.note]
        texts.extend(claim.detail for claim in self.claims)
        texts.extend(c.value for c in self.claims if isinstance(c.value, str))
        texts.extend(check.detail for check in self.checks)
        texts.extend(entry.reason for entry in self.cleared)
        texts.extend(entry.reason for entry in self.ruled_out)
        texts.extend(gap.detail for gap in self.coverage)
        return [t for t in texts if t]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "window": {
                "start": self.window_start,
                "end": self.window_end,
                "grain": self.grain,
            },
            "detector": self.detector,
            "mode": self.mode,
            "note": self.note,
            "headline": self.headline,
            "claims": [
                {
                    "key": c.key,
                    "label": c.label,
                    "value": _jsonable(c.value),
                    "unit": c.unit,
                    "display": c.display,
                    "detail": c.detail,
                }
                for c in self.claims
            ],
            "checks": [
                {
                    "name": c.name,
                    "state": c.state,
                    "score": _jsonable(c.score),
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "cleared": [
                {
                    "segment": e.segment,
                    "predicted": _jsonable(e.predicted),
                    "observed": _jsonable(e.observed),
                    "residual": _jsonable(e.residual),
                    "unit": e.unit,
                    "reason": e.reason,
                }
                for e in self.cleared
            ],
            "ruled_out": [
                {"segment": r.segment, "status": r.status, "reason": r.reason}
                for r in self.ruled_out
            ],
            "coverage": [
                {
                    "segment": g.segment,
                    "reason": g.reason,
                    "resolvable_effect": _jsonable(g.resolvable_effect),
                    "detail": g.detail,
                }
                for g in self.coverage
            ],
            "labels": list(self.labels),
        }


def _jsonable(value: Any) -> Any:
    """Replace values ``json.dumps`` emits as invalid JSON with null.

    A ratio against an empty denominator can reach here as inf or nan. ``json.dumps`` writes
    those as bare ``Infinity`` and ``NaN``, which is not JSON, and the parser on the other side
    of the deliverable rejects the whole document rather than the one field.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value) if math.isfinite(float(value)) else None


def _change_unit(level_unit: Unit) -> Unit:
    """The unit of a difference between two values of ``level_unit``.

    A difference of two proportions is percentage points, not percent, and conflating them is
    the error the unit system here exists to catch.
    """
    return POINTS if level_unit == PROBABILITY else level_unit


def _display(value: float | str, unit: Unit) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "not available"
    out = float(value)
    if unit == PERCENT:
        return f"{out * 100:.1f}%"
    if unit == POINTS:
        return f"{out * 100:+.2f}pp"
    if unit == PROBABILITY:
        # A p-value and a pass score are both probabilities, but rendering 1.2e-14 as "0.0%"
        # discards the only thing about it anybody cares about.
        if 0.0 < abs(out) < 0.01:
            return f"{out:.2g}"
        return f"{out * 100:.1f}%"
    if unit == CURRENCY:
        return f"{out:,.2f}"
    if unit == COUNT:
        return f"{out:,.0f}"
    return f"{out:.6g}"


_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

# Spans whose digits are timestamps rather than measurements. Masked before extraction because
# no amount of tolerance tuning makes "14:00" a figure that should resolve to a claim.
_TIMESTAMP_PATTERNS = (
    re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:,\s*\d{{4}})?\b", re.IGNORECASE),
)

_FIGURE = re.compile(
    r"""
    (?P<currency>[$\u20ac\u00a3\u20b9]\s?)?
    (?P<sign>[+-]\s?)?
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
      | \d+(?:\.\d+)?(?:[eE][+-]?\d+)?
      | \.\d+
    )
    (?P<unit>
        \s?%
      | \s?percentage\s+points?
      | \s?percent
      | \s?pp\b
      | x\b
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# A dimension name introducing a value, as `Segment.label()` renders it and as the statistics
# code embeds it mid-sentence: "Removing device_model=Galaxy S23 collapses the deviation to 12%".
_LABEL_KEY = re.compile(r"\b[a-z][a-z0-9_]*=")
_VALUE_TERMINATORS = ").,;:"


def _value_after_key(text: str, start: int) -> str:
    """Read one dimension value out of running prose, stopping where the value stops.

    Taking everything up to the next comma or bracket is what a first attempt does, and it is
    wrong in the way that matters: minimality writes "Removing device_model=Galaxy S23
    collapses the deviation to 12% of its original size", so a greedy read swallows the 12% and
    the verifier then treats a genuine figure as part of a product name and stops checking it.

    A value runs while its words could belong to a name -- they carry a digit or an uppercase
    letter -- and ends at the first ordinary lowercase word, at closing punctuation, or at the
    conjunction between two dimensions of the same segment.
    """
    words: list[str] = []
    for token in text[start:].split(" ")[:4]:
        if not token or token == "AND":
            break
        cleaned = token.rstrip(_VALUE_TERMINATORS)
        if not cleaned:
            break
        qualifies = any(ch.isdigit() for ch in cleaned) or any(ch.isupper() for ch in cleaned)
        if not qualifies:
            break
        words.append(cleaned)
        if cleaned != token:
            break
    return " ".join(words)


_MASK = "\x00"


def _label_vocabulary(*texts: str, segments: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Proper nouns whose digits must not be read as figures.

    Built from the bundle rather than from a list of device names, because the only strings a
    faithful narration can contain are ones the bundle handed it. A hard-coded "Android|iOS|
    Galaxy" list goes stale on the first dataset with different hardware in it, and a generic
    "capitalised word followed by a number" rule would excuse a fabricated figure in any
    sentence that happens to start with one -- which is exactly the hole the verifier exists
    to close, reopened in the name of convenience.

    A value made entirely of digits is deliberately excluded. Admitting "15" because some
    dimension is literally called 15 would excuse every 15 anywhere in the prose.
    """
    found: set[str] = set()
    for label in segments:
        for part in (label or "").split(" AND "):
            _, _, value = part.rpartition("=")
            found.add(value.strip().rstrip(_VALUE_TERMINATORS))
    for text in texts:
        if not text:
            continue
        for match in _LABEL_KEY.finditer(text):
            found.add(_value_after_key(text, match.end()))
    usable = {
        value
        for value in found
        if len(value) >= 2
        and any(ch.isalpha() for ch in value)
        and any(ch.isdigit() for ch in value)
    }
    return tuple(sorted(usable, key=len, reverse=True))


def _mask_non_figures(text: str, labels: tuple[str, ...]) -> str:
    """Blank out every span whose digits are not measurements, keeping offsets intact.

    Segment names go first, because a name can contain something that also parses as a date and
    the wrong one of the two would win. Timestamps follow. The third class -- digits welded to
    letters, "S23" and "2nd" -- is not handled here at all: `_is_standalone` settles those with
    a boundary rule that needs no vocabulary, and so keeps working on data this bundle has
    never seen.

    The replacement is a run of NULs rather than an empty string so that offsets stay aligned
    with the original text, which is what lets an unsupported literal be reported as the reader
    wrote it, and so that deleting a span cannot fuse the tokens on either side of it.
    """
    masked = text
    for label in labels:
        if not label:
            continue
        masked = re.sub(re.escape(label), lambda m: _MASK * len(m.group(0)), masked,
                        flags=re.IGNORECASE)
    for pattern in _TIMESTAMP_PATTERNS:
        masked = pattern.sub(lambda m: _MASK * len(m.group(0)), masked)
    return masked


@dataclass(frozen=True)
class _Literal:
    """A numeric token lifted out of prose, with everything needed to judge it.

    ``exact`` and ``rounded`` are both half a unit in the last place the literal was written
    at, differing only in whether the trailing zeros of an integer count as written. "275,600"
    is exact to within 0.5 read literally and to within 50 read as a figure rounded to the
    hundred, and both readings occur in real prose.
    """

    raw: str
    magnitude: float
    exact: float
    rounded: float
    unit: Unit


def _granularity(written: str) -> tuple[float, float]:
    """Half a unit in the last written place, taken literally and taken as significant figures.

    The exponent is folded in, so "6.8e-05" is faithful to anything within 5e-7 rather than
    within 0.05, which would otherwise accept almost any p-value at all.
    """
    mantissa, _, exponent = written.partition("e") if "e" in written else written.partition("E")
    power = int(exponent) if exponent else 0

    if "." in mantissa:
        step = 0.5 * (10.0 ** (power - len(mantissa.partition(".")[2])))
        return step, step

    exact = 0.5 * (10.0**power)
    digits = mantissa.lstrip("+-").lstrip("0")
    if not digits.strip("0"):
        return exact, exact
    trailing = len(digits) - len(digits.rstrip("0"))
    return exact, 0.5 * (10.0 ** (power + trailing))


def _parse_literal(match: re.Match[str]) -> _Literal | None:
    written = match.group("num").replace(",", "")
    try:
        value = float(written)
    except ValueError:
        return None

    exact, rounded = _granularity(written)
    suffix = (match.group("unit") or "").strip().lower()
    if suffix.startswith("%") or suffix.startswith("percent"):
        unit = POINTS if "point" in suffix else PERCENT
    elif suffix.startswith("pp"):
        unit = POINTS
    elif match.group("currency"):
        unit = CURRENCY
    else:
        unit = PLAIN
    return _Literal(match.group(0).strip(), abs(value), exact, rounded, unit)


def _is_standalone(text: str, match: re.Match[str]) -> bool:
    """Whether a numeric token is a figure rather than part of an identifier.

    Rejects "S23", "2nd", "4G" and "1.2.3" without needing to know what any of them are: a
    digit run welded to a letter is part of a name, and a digit run followed by another decimal
    group is part of a version. This is what keeps the allow-list small enough to stay honest.
    """
    start, end = match.span()
    if start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_."):
        return False
    tail = text[end:]
    if tail[:1].isalnum() or tail[:1] == "_":
        return False
    return not (tail[:1] == "." and tail[1:2].isdigit())


def _figures_in_text(text: str, labels: tuple[str, ...]) -> list[tuple[float, Unit]]:
    """Every figure a deterministically generated string states, in canonical form.

    Canonical means the same form a claim stores: proportions for percents and points, so that
    "99%" harvested out of a sufficiency explanation and a stored score of 0.99 are the same
    supported figure rather than two.
    """
    masked = _mask_non_figures(text, labels)
    out: list[tuple[float, Unit]] = []
    for match in _FIGURE.finditer(masked):
        if not _is_standalone(masked, match):
            continue
        literal = _parse_literal(match)
        if literal is None:
            continue
        if literal.unit in (PERCENT, POINTS):
            out.append((literal.magnitude / 100.0, literal.unit))
        else:
            out.append((literal.magnitude, literal.unit))
    return out


@dataclass(frozen=True)
class _Reading:
    """One thing a literal could denote, with the slack that reading allows."""

    value: float
    exact: float
    rounded: float
    units: frozenset[str] | None


def _readings(literal: _Literal) -> list[_Reading]:
    """What a written literal could denote, each with its slack and permitted claim units.

    The percent-versus-proportion rule lives here. A literal written "16.2%" is accepted
    against a stored 0.162 -- this bundle's own convention, where proportions are stored
    unscaled and rendered by `_display` -- and also against a stored 16.2, because a metric may
    declare a ``scale`` that puts the percentage into the stored value, and refusing that
    reading would flag faithful prose about any such metric. Accepting both is a deliberate
    widening and it costs something specific: a percentage that happens to equal an unrelated
    stored count passes. That is misattribution, which no numeral check can see; invention,
    which is what this catches, is unaffected.

    Every scaling divides the slack along with the value. Dividing a literal by a hundred but
    not its rounding tolerance would make "45%" faithful to anything within half a unit of
    0.45, which is every proportion in the bundle.
    """
    magnitude, exact, rounded = literal.magnitude, literal.exact, literal.rounded
    if literal.unit == PERCENT:
        return [
            _Reading(magnitude / 100.0, exact / 100.0, rounded / 100.0, _PERCENT_UNITS),
            _Reading(magnitude, exact, rounded, _PERCENT_UNITS),
        ]
    if literal.unit == POINTS:
        return [
            _Reading(magnitude / 100.0, exact / 100.0, rounded / 100.0, _POINT_UNITS),
            _Reading(magnitude, exact, rounded, _POINT_UNITS),
        ]
    if literal.unit == CURRENCY:
        return [_Reading(magnitude, exact, rounded, _CURRENCY_UNITS)]
    # A bare literal asserts nothing about units, so it may resolve to anything. The scaled
    # reading covers prose that writes a score as "93" rather than "0.93", and is restricted to
    # proportional claims so that it cannot reach a count or a currency amount.
    return [
        _Reading(magnitude, exact, rounded, None),
        _Reading(magnitude / 100.0, exact / 100.0, rounded / 100.0, _PROPORTIONAL_UNITS),
    ]


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    unsupported: list[str]
    checked: int


def verify_numbers(
    text: str, bundle: EvidenceBundle, *, tolerance: float = 0.05
) -> VerificationResult:
    """Confirm every figure in ``text`` was computed rather than written.

    A literal is supported when re-rounding some figure in the bundle to the precision the
    literal was written at reproduces the literal, under a reading its written form permits.
    "45%" is therefore a faithful rendering of a computed 44.79 and "16%" of 16.2, while "62%"
    for a computed 64.4 is not: the second decimal was not written, so it cannot be wrong, but
    the first was.

    ``tolerance`` bounds how far the significant-figures reading of a round number may stretch,
    as a fraction of the value. It is not a flat relative tolerance on every comparison, and
    the difference matters: five percent of a proportion is three percentage points, which is
    the width of a whole different finding.

    Two limits are worth stating rather than leaving to be discovered. Magnitudes are compared,
    not signs, because prose puts direction in the verb -- a sign-sensitive check would reject
    "fell 30%" for a stored -0.30, which is every honest sentence this system writes. And a
    figure that is in the bundle but attached to the wrong quantity passes, because no numeral
    check can see attribution. Direction and subject travel as text in the claim tuples and are
    what the unit rules and the template protect; this function catches invention, which is a
    different failure with a different remedy.
    """
    figures = bundle.figures()
    masked = _mask_non_figures(text, bundle.labels)
    years = _permitted_years(bundle)

    unsupported: list[str] = []
    checked = 0
    for match in _FIGURE.finditer(masked):
        if not _is_standalone(masked, match):
            continue
        literal = _parse_literal(match)
        if literal is None:
            continue
        checked += 1
        if _matches(literal, figures, tolerance):
            continue
        if _is_permitted_year(literal, years):
            continue
        unsupported.append(text[match.start() : match.end()].strip())

    return VerificationResult(not unsupported, unsupported, checked)


def _matches(
    literal: _Literal, figures: tuple[tuple[float, Unit], ...], tolerance: float
) -> bool:
    for reading in _readings(literal):
        for value, unit in figures:
            if reading.units is not None and unit not in reading.units:
                continue
            target = abs(value)
            # Slack is how far the value may sit from the literal and still round back to it.
            # The literal reading is the floor and the significant-figures reading the ceiling,
            # with `tolerance` deciding how much of the gap between them is allowed. Without
            # that cap, "20,000" would mean anything within five thousand; with a flat relative
            # tolerance instead, five percent of a proportion is three percentage points and
            # "62%" would verify against a computed 64.4%.
            slack = min(reading.rounded, max(reading.exact, tolerance * target))
            if abs(reading.value - target) <= slack + 1e-12:
                return True
    return False


def _permitted_years(bundle: EvidenceBundle) -> frozenset[int]:
    """Years a date in the prose may name.

    Restricted to the years the investigation actually touches rather than any plausible year,
    so that "2026" is excused in "late June 2026" while an invented count of 2026 requests in a
    bundle about 2024 is still caught. The year before the window is included because the
    baseline arms reach several weeks back and can cross a new year.
    """
    years: set[int] = set()
    for text in (bundle.window_start, bundle.window_end):
        for found in re.findall(r"\b(19\d{2}|20\d{2})\b", text or ""):
            years.add(int(found))
            years.add(int(found) - 1)
    return frozenset(years)


def _is_permitted_year(literal: _Literal, years: frozenset[int]) -> bool:
    if literal.unit != PLAIN or "." in literal.raw:
        return False
    return literal.magnitude.is_integer() and int(literal.magnitude) in years


def _metric_units(finding: Finding) -> tuple[Unit, Unit]:
    """Units for a level of this metric and for a change in it.

    Inferred from the statistical model rather than looked up in the registry, because a
    ``Localization`` carries the metric only by name and threading a ``MetricRegistry`` into
    the narration layer would make prose generation depend on config loading. ``two_proportion``
    is chosen by the detector exactly when the metric is a proportion, so the inference is
    exact for anything this pipeline produced, and a wrong guess costs a unit label rather than
    a figure.
    """
    model = getattr(getattr(finding, "test", None), "model", "") or ""
    if model == "two_proportion":
        return PROBABILITY, POINTS
    return PLAIN, PLAIN


def _direction_word(value: float | None) -> str:
    if value is None or abs(value) < 1e-12:
        return "held flat"
    return "rose" if value > 0 else "fell"


def _metric_phrase(metric: str) -> str:
    return metric.replace("_", " ")


def build_evidence(
    localization: Localization,
    finding: Finding,
    confidence: Any = None,
) -> EvidenceBundle:
    """Assemble every figure the narration layer may state, and nothing else.

    Counts of things -- candidates considered, checks run, weeks pooled -- are claims like any
    other. They look too trivial to bother with until a model writes "3 of the 4 tests held"
    and the verifier, having no claim to resolve "3" or "4" against, discards an otherwise
    perfect paragraph. Enumerations are figures; treating them as figures is what lets the
    verifier stay strict everywhere else.
    """
    window = localization.window
    level_unit, change_unit = _metric_units(finding)
    metric = _metric_phrase(localization.metric)
    accused = localization.accused

    claims: list[Claim] = []
    checks: list[EvidenceCheck] = []
    cleared: list[ExonerationEntry] = []
    ruled_out: list[RuledOut] = []
    coverage: list[CoverageEntry] = []

    window_label = window.label()
    claims.append(
        Claim(
            "window.duration_hours",
            "length of the window under investigation, in hours",
            window.duration.total_seconds() / 3600.0,
            COUNT,
            f"The window runs {window_label} at {window.grain} grain.",
        )
    )

    claims.extend(_parent_claims(localization, level_unit, change_unit, metric, window_label))
    claims.extend(_finding_claims(finding, level_unit, metric, window_label))

    if accused is not None:
        claims.extend(_accused_claims(accused, level_unit, change_unit, metric, window_label))
        for name in ("sufficiency", "minimality", "maximality", "holdout"):
            check = accused.checks.get(name)
            if check is None:
                continue
            checks.append(EvidenceCheck(name, check.state, check.score, check.detail))
            if check.state == "unknown":
                coverage.append(
                    CoverageEntry(
                        accused.segment.label(),
                        f"{name}_untested",
                        None,
                        check.detail,
                    )
                )

    for candidate in localization.cleared:
        cleared.append(
            ExonerationEntry(
                candidate.segment.label(),
                candidate.predicted_if_innocent,
                candidate.observed_value,
                candidate.exoneration_residual,
                level_unit,
                candidate.reason,
            )
        )
    for candidate in localization.runners_up:
        ruled_out.append(
            RuledOut(candidate.segment.label(), candidate.status, candidate.reason)
        )

    coverage.extend(_coverage_from_candidates(localization, accused))
    if finding.resolvable_effect is None:
        coverage.append(
            CoverageEntry(
                finding.segment.label(),
                "sensitivity_not_computed",
                None,
                "No smallest-resolvable-move figure was computed for this cell, so how small a "
                "change it could have detected is unknown rather than known to be adequate.",
            )
        )

    claims.extend(
        _tally_claims(localization, checks, coverage, cleared, ruled_out, window_label)
    )
    claims.extend(_confidence_claims(confidence))

    labels = _label_vocabulary(
        *[c.detail for c in claims],
        *[c.detail for c in checks],
        *[e.reason for e in cleared],
        *[r.reason for r in ruled_out],
        *[g.detail for g in coverage],
        segments=tuple(
            [localization.parent.label(), finding.segment.label()]
            + [c.segment.label() for c in localization.candidates]
            + [e.segment for e in cleared]
            + [r.segment for r in ruled_out]
            + [g.segment for g in coverage]
        ),
    )

    headline = _headline(localization, finding, level_unit, metric, window_label)

    return EvidenceBundle(
        metric=localization.metric,
        window_start=window.start.isoformat(),
        window_end=window.end.isoformat(),
        grain=window.grain,
        detector=finding.detector,
        headline=headline,
        claims=claims,
        checks=checks,
        cleared=cleared,
        ruled_out=ruled_out,
        coverage=coverage,
        labels=labels,
        mode=localization.mode,
        note=localization.note,
    )


def _parent_claims(
    localization: Localization,
    level_unit: Unit,
    change_unit: Unit,
    metric: str,
    window_label: str,
) -> list[Claim]:
    parent = localization.parent.label()
    provenance = (
        f"Measured over {parent} for {window_label} against the pooled same-weekday baseline "
        "weeks."
    )
    out = [
        Claim("parent.segment", "population the movement was measured over", parent, PLAIN,
              "The top of the lattice; every candidate is a subset of it."),
        Claim("parent.deviation", f"change in {metric} across {parent}",
              localization.parent_deviation, change_unit, provenance),
    ]
    if localization.parent_observed is not None:
        out.append(
            Claim("parent.observed", f"{metric} across {parent} in the window",
                  localization.parent_observed, level_unit, provenance)
        )
    if localization.parent_expected is not None:
        out.append(
            Claim("parent.expected", f"expected {metric} across {parent}",
                  localization.parent_expected, level_unit, provenance)
        )
    return out


def _finding_claims(
    finding: Finding, level_unit: Unit, metric: str, window_label: str
) -> list[Claim]:
    test = finding.test
    segment = finding.segment.label()
    provenance = (
        f"Produced by the {finding.detector} detector's {test.model} test on {segment} over "
        f"{window_label}."
    )
    out = [
        Claim("finding.segment", "segment the detector flagged", segment, PLAIN, provenance),
        Claim("finding.direction", f"direction the {metric} moved in",
              _direction_word(test.relative_effect), PLAIN, provenance),
        Claim("finding.relative_effect", f"relative change in {metric} for {segment}",
              test.relative_effect, PERCENT, provenance),
        Claim("finding.observed", f"{metric} for {segment} in the window", test.observed,
              level_unit, provenance),
        Claim("finding.expected", f"expected {metric} for {segment}", test.expected,
              level_unit, provenance),
        Claim("finding.z", "standardised deviation of the test statistic", test.z, PLAIN,
              provenance),
        Claim("finding.p_value", "probability of a deviation this large under the null",
              test.p_value, PROBABILITY,
              provenance + " Compared after Benjamini-Hochberg correction over the whole "
              "family of tested cells."),
        Claim("finding.dispersion", "overdispersion factor applied to the variance",
              finding.phi, PLAIN,
              "Estimated robustly from the historical arms only, so the incident under "
              "investigation cannot inflate the variance used to judge it."),
        Claim("finding.effect_threshold", "smallest change considered worth reporting",
              finding.effect_threshold, PERCENT,
              "A reporting policy for this metric, not a detection limit."),
        Claim("baseline.weeks_kept", "baseline weeks pooled into the expectation",
              finding.weeks_kept, COUNT,
              f"Out of {finding.weeks_seen} aligned weeks available; the rest were dropped as "
              "unusable or trimmed as extreme."),
        Claim("finding.screening", "how this finding was screened",
              _SCREENING_LABEL.get(finding.screening, "not screened"), PLAIN,
              "Benjamini-Hochberg runs over the temporal family only. A structural residual "
              "is screened against a fixed threshold instead, because pooling the two would "
              "mix a weekly-baseline null with a median-polish null."),
    ]
    if finding.resolvable_effect is not None:
        out.append(
            Claim("finding.resolvable_effect",
                  "smallest relative move this cell could have resolved",
                  finding.resolvable_effect, PERCENT,
                  "Computed from the traffic the cell actually carried, so a result can be "
                  "told apart from one sitting at the edge of what the data supports.")
        )
    return out


def _accused_claims(
    accused: Candidate,
    level_unit: Unit,
    change_unit: Unit,
    metric: str,
    window_label: str,
) -> list[Claim]:
    segment = accused.segment.label()
    provenance = f"Measured on {segment} over {window_label} against its own pooled baseline."
    out = [
        Claim("accused.segment", "segment held responsible", segment, PLAIN,
              "Chosen by counterfactual removal, not by the size of its movement."),
        Claim("accused.relative_effect", f"{metric} change for {segment}",
              accused.relative_effect, PERCENT, provenance),
        Claim("accused.deviation", f"absolute {metric} change for {segment}",
              accused.deviation, change_unit, provenance),
        Claim("accused.sufficiency", "share of the parent movement this segment accounts for",
              accused.sufficiency, PROBABILITY,
              "From the sufficiency counterfactual: the parent metric recomputed with this "
              "segment removed."),
        Claim("accused.requests", "requests this segment carried in the window",
              accused.observed.requests, COUNT, provenance),
    ]
    if accused.observed_value is not None:
        out.append(
            Claim("accused.observed", f"{metric} for {segment} in the window",
                  accused.observed_value, level_unit, provenance)
        )
    if accused.expected_value is not None:
        out.append(
            Claim("accused.expected", f"expected {metric} for {segment}",
                  accused.expected_value, level_unit, provenance)
        )
    if accused.observed.revenue:
        out.append(
            Claim("accused.revenue", "revenue this segment carried in the window",
                  accused.observed.revenue, CURRENCY, provenance)
        )
    return out


def _coverage_from_candidates(
    localization: Localization, accused: Candidate | None
) -> list[CoverageEntry]:
    """Candidates the ledger could not speak about at all.

    A candidate with no predicted value was neither accused nor cleared: no rollup cell covers
    its overlap with the accused, so there is no arithmetic that could exonerate it. Leaving it
    out of the case file entirely would present a partial ledger as a complete one.
    """
    out: list[CoverageEntry] = []
    if accused is None:
        return out
    for candidate in localization.candidates:
        if candidate is accused or candidate.status in {"cleared", "accused"}:
            continue
        if candidate.predicted_if_innocent is not None:
            continue
        out.append(
            CoverageEntry(
                candidate.segment.label(),
                "not_cleared_quantitatively",
                None,
                candidate.reason
                or "No rollup cell covers the overlap between this segment and the accused, so "
                "no prediction could be made for it.",
            )
        )
    return out


def _tally_claims(
    localization: Localization,
    checks: list[EvidenceCheck],
    coverage: list[CoverageEntry],
    cleared: list[ExonerationEntry],
    ruled_out: list[RuledOut],
    window_label: str,
) -> list[Claim]:
    provenance = f"Counted over the candidate set this localization built for {window_label}."
    return [
        Claim("candidates.considered", "candidate segments tested",
              len(localization.candidates), COUNT, provenance),
        Claim("candidates.cleared", "candidates cleared by prediction", len(cleared), COUNT,
              "Each one has a predicted and an observed value in the exoneration ledger."),
        Claim("candidates.ruled_out", "candidates rejected as too broad, too narrow, or partial",
              len(ruled_out), COUNT, provenance),
        Claim("checks.run", "counterfactual tests run on the accused segment", len(checks),
              COUNT, provenance),
        Claim("checks.passed", "counterfactual tests the accused segment passed",
              sum(1 for c in checks if c.state == "pass"), COUNT, provenance),
        Claim("checks.failed", "counterfactual tests the accused segment failed",
              sum(1 for c in checks if c.state == "fail"), COUNT, provenance),
        Claim("coverage.gaps", "questions that could not be answered with the data available",
              len(coverage), COUNT,
              "Recorded rather than dropped, so that silence about a segment is never mistaken "
              "for evidence of its innocence."),
    ]


def _confidence_claims(confidence: Any) -> list[Claim]:
    """Read the confidence score defensively.

    Typed loosely and reached through ``getattr`` on purpose. The scorer is a separate module
    with its own shape, and narration should degrade to omitting a paragraph if that shape
    changes rather than raising in the middle of writing a case file whose figures are already
    correct.
    """
    if confidence is None:
        return []

    out: list[Claim] = []
    value = getattr(confidence, "value", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(
            Claim("confidence.value", "overall confidence in this verdict", float(value),
                  PROBABILITY,
                  "A weighted combination of the components below, each computed from a test "
                  "that was actually run.")
        )

    components = getattr(confidence, "components", None)
    if not isinstance(components, (list, tuple)):
        components = ()
    for component in components:
        name = str(getattr(component, "name", "") or "component")
        detail = str(getattr(component, "detail", "") or "")
        state = str(getattr(component, "state", "") or "unknown")
        score = getattr(component, "score", None)
        weight = getattr(component, "weight", None)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            out.append(
                Claim(f"confidence.{name}.score", f"{name} component of the confidence score",
                      float(score), PROBABILITY, detail or f"Component state: {state}.")
            )
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            out.append(
                Claim(f"confidence.{name}.weight", f"weight given to the {name} component",
                      float(weight), PROBABILITY,
                      f"Fixed by configuration, not fitted to this case. Component state: "
                      f"{state}.")
            )

    # Read by name and skipped when absent, like everything else here. These are the scorer's
    # own coverage figures, and prose that says "2 of the 5 components could be scored" needs
    # claims to resolve those digits against or an otherwise faithful sentence is discarded.
    for attribute, label in (
        ("components_scored", "confidence components that could be scored"),
        ("components_total", "confidence components defined"),
    ):
        tally = getattr(confidence, attribute, None)
        if isinstance(tally, int) and not isinstance(tally, bool):
            out.append(
                Claim(f"confidence.{attribute}", label, tally, COUNT,
                      "A component with no test behind it is left unscored rather than "
                      "assigned a neutral value that would dilute the ones that ran.")
            )

    publishable = getattr(confidence, "publishable", None)
    if isinstance(publishable, bool):
        out.append(
            Claim("confidence.publishable", "clears the publication threshold",
                  "yes" if publishable else "no", PLAIN,
                  "Compared against the configured threshold, not chosen per case.")
        )

    caveat = getattr(confidence, "caveat", None)
    if isinstance(caveat, str) and caveat.strip():
        out.append(
            Claim("confidence.caveat", "what the confidence score does not cover",
                  caveat.strip(), PLAIN, "Recorded by the confidence scorer.")
        )
    return out


def _headline(
    localization: Localization,
    finding: Finding,
    level_unit: Unit,
    metric: str,
    window_label: str,
) -> str:
    if localization.mode == "no_data":
        return (
            f"No {metric} rollup data was available for {window_label}, so no segment was "
            "tested and no verdict is offered."
        )

    accused = localization.accused
    if accused is None:
        total = len(localization.candidates)
        return (
            f"No segment could be held responsible for the {metric} movement in the window "
            f"{window_label}. Every one of the {_display(total, COUNT)} "
            f"{_plural(total, 'candidate', 'candidates')} tested failed at least one "
            "counterfactual test, and naming the best of them would be a guess dressed as a "
            "finding."
        )

    direction = _direction_word(accused.relative_effect)
    sentence = (
        f"{metric.capitalize()} for {accused.segment.label()} {direction} "
        f"{_display(abs(accused.relative_effect), PERCENT)} in the window {window_label}"
    )
    if accused.expected_value is not None and accused.observed_value is not None:
        sentence += (
            f", from an expected {_display(accused.expected_value, level_unit)} to an observed "
            f"{_display(accused.observed_value, level_unit)}"
        )
    return sentence + "."


def template_narration(bundle: EvidenceBundle) -> str:
    """Prose built from the bundle alone, with no model involved.

    This is the fallback, and it is also the reference: it is what the guarantee "turn the
    model off and only the prose degrades" actually means in practice. It quotes the
    statistics code's own explanations verbatim rather than paraphrasing them, because those
    sentences were written next to the arithmetic that produced them and are the least likely
    thing in the system to drift away from what was computed.
    """
    blocks = [bundle.headline]
    for heading, section in (
        ("What moved", _template_movement),
        ("Why this segment", _template_counterfactual),
        ("What was ruled out", _template_exoneration),
        ("What could not be tested", _template_coverage),
        ("Confidence", _template_confidence),
    ):
        text = section(bundle)
        if text:
            blocks.append(f"## {heading}\n{text}")
    return "\n\n".join(blocks)


def _bullets(lines: Sequence[str]) -> str:
    """One finding per line.

    These sections used to be joined with a space into paragraphs, and the result was a
    seventeen-line block in which the sufficiency test, the maximality test and the holdout
    test were three sentences deep in the same run of text. Each of those is a separate
    question with a separate answer, and a reader scanning for the one that failed should not
    have to read the other two to find it. The leading dash is safe against the numeric
    verifier, which compares magnitudes rather than signs.
    """
    return "\n".join(f"- {line}" for line in lines if line)


def _claim_map(bundle: EvidenceBundle) -> dict[str, Claim]:
    return {claim.key: claim for claim in bundle.claims}


def _plural(count: int, singular: str, plural: str) -> str:
    """Agreement for a count that is a claim.

    The count itself has to be rendered from the claim rather than from ``count`` so that the
    figure in the prose and the figure in the evidence file are the same object; only the noun
    is chosen here.
    """
    return singular if count == 1 else plural


def _shown(claims: dict[str, Claim], key: str) -> str | None:
    claim = claims.get(key)
    return claim.display if claim is not None else None


def _template_movement(bundle: EvidenceBundle) -> str:
    claims = _claim_map(bundle)
    parent = _shown(claims, "parent.segment") or "all traffic"
    metric = _metric_phrase(bundle.metric)

    sentences: list[str] = []
    observed = _shown(claims, "parent.observed")
    expected = _shown(claims, "parent.expected")
    deviation = _shown(claims, "parent.deviation")
    if observed and expected:
        sentences.append(
            f"Across {parent}, {metric} read {observed} against an expected {expected}, a "
            f"movement of {deviation}."
        )

    flagged = _shown(claims, "finding.segment")
    z = _shown(claims, "finding.z")
    p = _shown(claims, "finding.p_value")
    if flagged and z and p:
        sentences.append(
            f"The {bundle.detector} detector flagged {flagged} at z = {z} and p = {p}, with an "
            f"overdispersion factor of {_shown(claims, 'finding.dispersion')} applied to the "
            f"variance and the baseline pooled over "
            f"{_shown(claims, 'baseline.weeks_kept')} aligned weeks of the same weekday and "
            "hours."
        )

    resolvable = _shown(claims, "finding.resolvable_effect")
    threshold = _shown(claims, "finding.effect_threshold")
    if resolvable and threshold:
        sentences.append(
            f"On the traffic this cell carried, the smallest move it could have resolved is "
            f"{resolvable}, against a reporting threshold of {threshold}, so the result is not "
            "sitting at the edge of what the data supports."
        )

    if bundle.mode == "structural_only" and bundle.note:
        sentences.append(bundle.note)
    return _bullets(sentences)


def _template_counterfactual(bundle: EvidenceBundle) -> str:
    claims = _claim_map(bundle)
    accused = _shown(claims, "accused.segment")
    if accused is None:
        if not bundle.ruled_out:
            return ""
        lines = [
            "Every candidate was rejected for a stated reason rather than by omission.",
        ]
        for entry in bundle.ruled_out[:4]:
            lines.append(f"{entry.segment} was recorded as {entry.status}: {entry.reason}")
        return _bullets(lines)

    if bundle.mode == "structural_only":
        # Saying the verdict rests on removing the segment from a parent that never moved
        # contradicts the paragraph above and describes a test that did not run. In this mode
        # the breadth checks are the whole of the evidence and the prose has to say so.
        opening = (
            f"With the parent metric flat there is nothing for a counterfactual to restore, so "
            f"{accused} stands on its own deviation and on how its neighbours behaved rather "
            "than on explaining a movement in the total."
        )
    else:
        opening = (
            "The accusation rests on a counterfactual rather than on the size of the movement. "
            f"Removing {accused} from the population and recomputing the parent metric is what "
            "separates a cause from a segment that merely moved with it."
        )
    verbs = {"pass": "held", "fail": "failed", "unknown": "could not be run"}
    lines = [opening]
    for check in bundle.checks:
        if check.state == "unknown":
            continue
        lines.append(f"The {check.name} test {verbs.get(check.state, check.state)}: "
                     f"{check.detail}")
    return _bullets(lines)


def _template_exoneration(bundle: EvidenceBundle) -> str:
    if not bundle.cleared:
        return ""
    claims = _claim_map(bundle)
    count = _shown(claims, "candidates.cleared") or str(len(bundle.cleared))
    considered = _shown(claims, "candidates.considered") or str(len(bundle.cleared))
    lines = [
        f"Of the {considered} candidates tested, {count} "
        f"{_plural(len(bundle.cleared), 'was', 'were')} cleared by prediction rather than by "
        "omission: the ledger states what each one would have read if the accused segment "
        "explains everything, and what it actually read."
    ]
    for entry in bundle.cleared[:3]:
        predicted = _display(entry.predicted, entry.unit) if entry.predicted is not None else None
        observed = _display(entry.observed, entry.unit) if entry.observed is not None else None
        if predicted is None or observed is None:
            continue
        residual = (
            _display(entry.residual, _change_unit(entry.unit))
            if entry.residual is not None
            else "not available"
        )
        lines.append(
            f"{entry.segment} was predicted at {predicted} and read {observed}, a residual of "
            f"{residual}."
        )
    if len(bundle.cleared) > 3:
        lines.append("The remaining cleared segments are listed in the evidence file with the "
                     "same predicted and observed pair, so any of them can be checked by hand.")
    return _bullets(lines)


def _template_coverage(bundle: EvidenceBundle) -> str:
    claims = _claim_map(bundle)
    if not bundle.coverage:
        if "accused.segment" not in claims:
            return ""
        return (
            "Every test the lattice can express was run on the accused segment, so nothing "
            "here is being reported as untested."
        )

    count = _shown(claims, "coverage.gaps") or str(len(bundle.coverage))
    n = len(bundle.coverage)
    lines = [
        f"{count} {_plural(n, 'question', 'questions')} could not be answered with the data "
        f"available, and {_plural(n, 'is', 'are')} recorded here rather than dropped, because "
        "a segment nobody could test is not a segment that was found innocent."
    ]
    for gap in bundle.coverage[:3]:
        lines.append(f"On {gap.segment} -- {gap.detail}")
    return _bullets(lines)


def _template_confidence(bundle: EvidenceBundle) -> str:
    claims = _claim_map(bundle)
    value = _shown(claims, "confidence.value")
    if value is None:
        return ""
    parts = [f"Overall confidence in this verdict is {value}."]
    components = [
        claim for key, claim in claims.items()
        if key.startswith("confidence.") and key.endswith(".score")
    ]
    for claim in components[:4]:
        parts.append(f"The {claim.key.split('.')[1]} component scores {claim.display}.")
    caveat = claims.get("confidence.caveat")
    if caveat is not None and isinstance(caveat.value, str):
        parts.append(caveat.value)
    return _bullets(parts)


_SYSTEM_PROMPT = (
    "You write the summary of an automated root-cause investigation for an on-call engineer.\n"
    "\n"
    "Every figure has already been computed and checked by statistical code. You are given "
    "those figures as a list of claims. Your only job is to phrase them well.\n"
    "\n"
    "Rules, all of which are checked mechanically after you answer:\n"
    "1. State no number that is not the display value of one of the claims. Copy display "
    "values exactly as they are written, including the unit.\n"
    "2. Do not compute, add, average, convert, or re-round anything. If you want a figure that "
    "is not in the list, describe it in words instead.\n"
    "3. Never convert between percent and percentage points. They are different units and each "
    "claim tells you which one it is.\n"
    "4. Keep the direction of every change exactly as the claim states it.\n"
    "5. Quote segment names verbatim. Do not invent one and do not attach a figure to a "
    "segment other than the one whose claim carries it.\n"
    "6. Structure the answer so it can be scanned, not read start to finish. On-call engineers\n"
    "read these while something is broken; a wall of prose buries the one line that matters.\n"
    "Use exactly this shape:\n"
    "\n"
    "    One sentence stating the verdict. No heading above it.\n"
    "\n"
    "    ## What moved\n"
    "    - one fact per line: the segment, the figures, and which detector flagged it\n"
    "\n"
    "    ## Why this segment\n"
    "    - one test per line: name it, say whether it held or failed, and give the reason the\n"
    "      test itself supplied, with its figures\n"
    "\n"
    "    ## What was ruled out\n"
    "    - one cleared candidate per line, with its predicted and observed figures. Include a\n"
    "      candidate here only if both figures are given to you. A candidate with no figures\n"
    "      was not cleared by prediction and belongs in the section below, not in this one.\n"
    "\n"
    "    ## What could not be tested\n"
    "    - one gap per line, saying what could not be established about that segment\n"
    "\n"
    "    ## Confidence\n"
    "    - the overall figure, then one component per line\n"
    "\n"
    "7. Every bullet must carry its substance, not a label for it. \"The sufficiency test\n"
    "failed\" on its own is useless to a reader deciding what to do next; the input gives you\n"
    "the explanation each test produced, and the bullet has to carry it. Never write a bare\n"
    "reason code such as `not_cleared_quantitatively` or `minimality_untested` -- say what it\n"
    "means in words. A bullet is one sentence, not one phrase.\n"
    "\n"
    "8. Omit any section with nothing to report rather than writing that it is empty. The only\n"
    "markup permitted is `## ` for a heading and `- ` for a bullet: no bold, no italics, no\n"
    "tables, no numbered lists.\n"
    "\n"
    "A draft containing one figure that is not in the claim list is discarded in full and "
    "replaced by a template, so an unsupported number costs the entire answer."
)


def build_prompt(bundle: EvidenceBundle) -> str:
    """Render the claim tuples the model is allowed to speak from.

    Deliberately the only view of the investigation the model gets. It sees no rows, no
    counters it could divide, and no intermediate quantities -- there is nothing here to
    compute a new number from even if the instructions were ignored.
    """
    lines = [
        f"Metric: {bundle.metric}",
        f"Window: {bundle.window_start} to {bundle.window_end} at {bundle.grain} grain",
        f"Detector: {bundle.detector}",
        f"Localization mode: {bundle.mode}",
        "",
        "One-sentence summary already written by the deterministic template:",
        bundle.headline,
        "",
        "Claims. Use only these figures, written exactly as the display column shows them.",
    ]
    for claim in bundle.claims:
        lines.append(f"- {claim.key} | {claim.label} | {claim.display} | {claim.detail}")

    if bundle.checks:
        lines.extend(["", "Counterfactual tests, with the explanation each one produced:"])
        for check in bundle.checks:
            lines.append(f"- {check.name} [{check.state}] {check.detail}")

    if bundle.cleared:
        lines.extend(["", "Exoneration ledger, predicted against observed:"])
        for entry in bundle.cleared[:8]:
            lines.append(
                f"- {entry.segment} | predicted {_display(entry.predicted, entry.unit)} | "
                f"observed {_display(entry.observed, entry.unit)} | residual "
                f"{_display(entry.residual, _change_unit(entry.unit))}"
            )

    if bundle.ruled_out:
        lines.extend(["", "Candidates rejected by a breadth test:"])
        for entry in bundle.ruled_out[:8]:
            lines.append(f"- {entry.segment} [{entry.status}] {entry.reason}")

    if bundle.coverage:
        lines.extend(["", "What could not be tested:"])
        for gap in bundle.coverage[:8]:
            lines.append(f"- {gap.segment} [{gap.reason}] {gap.detail}")

    if bundle.note:
        lines.extend(["", f"Note from the localizer: {bundle.note}"])

    return "\n".join(lines)


@dataclass(frozen=True)
class Narration:
    text: str
    source: str
    verified: bool
    unsupported: list[str] = field(default_factory=list)
    model: str = ""
    # Carried so the case file can show what the model cost and how long it took, including on
    # the paths where its draft was thrown away. A discarded draft is the most interesting one:
    # it is the guardrail firing, and without a record it is a claim rather than a demonstration.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


def narrate(bundle: EvidenceBundle, cfg: LLMConfig, client: Any = None) -> Narration:
    """Phrase the bundle, preferring the model but never depending on it.

    The order matters. Verification runs on the draft before anything is returned, and a draft
    that fails is discarded whole rather than patched, because a paragraph with one figure
    removed is a paragraph whose remaining sentences were written to support it. The literals
    that failed are carried on the result so the fallback is recorded in the case file instead
    of looking like a run where the model was simply switched off.
    """
    template = template_narration(bundle)

    if not getattr(cfg, "enabled", False):
        return Narration(template, "template", False, [], "")

    if client is None:
        try:
            client = LLMClient(cfg)
        except Exception as exc:
            log.warning("Could not build a narration client: %s", exc)
            return Narration(template, "template", False, [], "")

    if not getattr(client, "available", False):
        return Narration(template, "template", False, [], "")

    try:
        completion = client.complete(_SYSTEM_PROMPT, build_prompt(bundle))
    except Exception as exc:
        # The client's contract is that it never raises, and the real one honours it. This
        # catch is here because narration is the last step before a finished case file is
        # returned, and a fake, a wrapper, or a future client that breaks that contract should
        # cost the prose rather than the investigation.
        log.warning("Narration client raised despite its contract: %s", exc)
        return Narration(template, "template", False, [], "")

    if not isinstance(completion, Completion) or not completion.ok or not completion.text.strip():
        # Still record the cost. A call that failed after retries, or one whose draft was cut off
        # at the token ceiling, consumed budget and time, and a case that silently reads
        # "template" gives no hint that a model was tried at all.
        return Narration(
            template, "template", False, [],
            getattr(completion, "model", "") if isinstance(completion, Completion) else "",
            getattr(completion, "prompt_tokens", 0) if isinstance(completion, Completion) else 0,
            getattr(completion, "completion_tokens", 0)
            if isinstance(completion, Completion) else 0,
            getattr(completion, "latency_ms", 0) if isinstance(completion, Completion) else 0,
        )

    result = verify_numbers(completion.text, bundle)
    if not result.ok:
        log.warning(
            "Discarded a narration draft carrying %d unsupported figure(s): %s",
            len(result.unsupported),
            ", ".join(result.unsupported),
        )
        return Narration(
            template, "template", False, result.unsupported, completion.model,
            completion.prompt_tokens, completion.completion_tokens, completion.latency_ms,
        )

    return Narration(
        completion.text.strip(), "llm", True, [], completion.model,
        completion.prompt_tokens, completion.completion_tokens, completion.latency_ms,
    )
