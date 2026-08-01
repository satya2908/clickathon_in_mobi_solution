"""Tests for the narration layer.

The suite is built around one property: the figures in a case file do not depend on the model.
Everything else here exists to defend that property against the two ways it fails in practice
-- a model that invents a number and is believed, and a verifier so literal that it rejects
faithful prose and quietly turns the model into decoration.

No network. Every client is a fake, and the fixtures are hand-built rather than imported from
the other suites, which are being edited alongside this one.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

from verdict.config import LLMConfig
from verdict.detect import Finding
from verdict.llm import Completion, LLMClient
from verdict.localize import Candidate, Check, Localization
from verdict.narrate import (
    EvidenceBundle,
    build_evidence,
    build_prompt,
    narrate,
    template_narration,
    verify_numbers,
)
from verdict.query import Counters, Segment, Window
from verdict.stats import TestResult

WINDOW = Window(datetime(2026, 6, 23), datetime(2026, 6, 26), "1d")

# The planted incident, sized so that the interesting arithmetic is distinguishable. Android 15
# fill rate falls from 0.785 to 0.4334, which is a relative move of -44.79% and an absolute move
# of -35.16 percentage points. Those two numbers are the percent/percentage-point pair the
# verifier has to keep apart, and 44.79 is the value a model rounds to "45%".
BASE_RATE = 0.785
BAD_RATE = 0.4334
RELATIVE_EFFECT = -0.4479
ABSOLUTE_EFFECT = -0.3516

PARENT_OBSERVED = 0.71468
PARENT_DEVIATION = -0.07032


def _check(name: str, state: str, score: float | None, detail: str) -> Check:
    return Check(name, state, score, detail)


def _accused() -> Candidate:
    candidate = Candidate(
        segment=Segment.of(os_version="Android 15"),
        observed=Counters(requests=20_000, fills=8_668, impressions=7_900, clicks=158,
                          revenue=275_597.12),
        expected=Counters(requests=20_000, fills=15_700, impressions=14_300, clicks=286,
                          revenue=498_120.44),
        observed_value=BAD_RATE,
        expected_value=BASE_RATE,
        status="accused",
    )
    candidate.checks = {
        "sufficiency": _check(
            "sufficiency", "pass", 0.93,
            "Parent moved -0.0703. With os_version=Android 15 removed it moves -0.0049, so 93% "
            "of the change is accounted for by this segment.",
        ),
        "minimality": _check(
            "minimality", "pass", 0.71,
            "Removing the guiltiest child (device_model=Galaxy S23 AND os_version=Android 15) "
            "still leaves 71% of the deviation, so the cause is spread across this segment "
            "rather than concentrated in one part of it.",
        ),
        "maximality": _check(
            "maximality", "pass", 0.88,
            "Only 12% of the 8 sibling values of os_version moved with it, so the deviation "
            "really is specific to this candidate.",
        ),
        "holdout": _check(
            "holdout", "pass", 0.82,
            "First half -47.9%, second half -41.5%. The effect reproduces on the half it was "
            "not selected on.",
        ),
    }
    return candidate


def _cleared(segment: Segment, predicted: float, observed: float) -> Candidate:
    residual = observed - predicted
    candidate = Candidate(
        segment=segment,
        observed=Counters(requests=15_000, fills=int(15_000 * observed)),
        expected=Counters(requests=15_000, fills=int(15_000 * BASE_RATE)),
        observed_value=observed,
        expected_value=BASE_RATE,
        predicted_if_innocent=predicted,
        exoneration_residual=residual,
        status="cleared",
    )
    candidate.reason = (
        f"Predicted {predicted:.6g} if the accused explains everything; observed "
        f"{observed:.6g}. Residual {residual:+.6g} is within tolerance, so this segment's "
        "movement is fully accounted for."
    )
    return candidate


def _partial() -> Candidate:
    candidate = Candidate(
        segment=Segment.of(ad_format="banner", country="AR"),
        observed=Counters(requests=4_000, fills=2_900),
        expected=Counters(requests=4_000, fills=3_140),
        observed_value=0.725,
        expected_value=BASE_RATE,
        status="partial",
    )
    candidate.reason = (
        "Could not be cleared quantitatively: no rollup cell covers the overlap between this "
        "segment and the accused."
    )
    return candidate


def _finding(segment: Segment | None = None) -> Finding:
    return Finding(
        metric="fill_rate",
        segment=segment or Segment.of(os_version="Android 15"),
        window=WINDOW,
        detector="temporal",
        test=TestResult(
            z=-42.13,
            p_value=1.2e-14,
            observed=BAD_RATE,
            expected=BASE_RATE,
            absolute_effect=ABSOLUTE_EFFECT,
            relative_effect=RELATIVE_EFFECT,
            model="two_proportion",
        ),
        observed_counters=Counters(requests=20_000, fills=8_668),
        baseline_counters=Counters(requests=20_000, fills=15_700),
        phi=1.87,
        weeks_kept=3,
        weeks_seen=4,
        survives_correction=True,
        effect_threshold=0.05,
        resolvable_effect=0.012,
    )


def _localization() -> Localization:
    accused = _accused()
    candidates = [
        accused,
        _cleared(Segment.of(device_model="Galaxy S23"), 0.6440, 0.6443),
        _cleared(Segment.of(device_model="iPhone 14"), 0.7834, 0.7831),
        _cleared(Segment.of(os_version="iOS 17.2"), 0.7850, 0.7846),
        _partial(),
    ]
    return Localization(
        metric="fill_rate",
        window=WINDOW,
        parent=Segment.total(),
        parent_observed=PARENT_OBSERVED,
        parent_expected=BASE_RATE,
        parent_deviation=PARENT_DEVIATION,
        accused=accused,
        candidates=candidates,
        mode="explain_away",
    )


@pytest.fixture
def bundle() -> EvidenceBundle:
    return build_evidence(_localization(), _finding())


OFF = LLMConfig(enabled=False)
ON = LLMConfig(enabled=True, api_key="not-a-real-key", model="fake-model-1")


class FakeClient:
    """Returns a canned draft. Nothing here touches a socket."""

    available = True

    def __init__(self, text: str, *, ok: bool = True, model: str = "fake-model-1") -> None:
        self.text = text
        self.ok = ok
        self.model = model
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        return Completion(self.text, self.model, self.ok, "" if self.ok else "simulated failure")


class ExplodingClient:
    available = True

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def complete(self, system: str, user: str) -> Completion:
        raise self.exc


class UnavailableClient:
    available = False

    def complete(self, system: str, user: str) -> Completion:  # pragma: no cover - never called
        raise AssertionError("An unavailable client must never be called.")


class ReorderingClient:
    """A faithful paraphrase in the only sense that matters: same figures, different prose.

    Rewrites the template by reversing its sentences, which changes every line of the text
    while provably preserving every numeral in it. A hand-written paraphrase would test the
    same path but would silently stop testing it the day the template changes wording.
    """

    available = True

    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle

    def complete(self, system: str, user: str) -> Completion:
        sentences = [s.strip() for s in template_narration(self.bundle).split(". ") if s.strip()]
        return Completion(". ".join(reversed(sentences)), "fake-model-1", True, "")


_NUMERAL = re.compile(r"\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")


def numerals(text: str) -> set[str]:
    """Every numeral-shaped token in a piece of prose, extracted naively on purpose.

    Deliberately not the module's own extractor. The point of comparing two outputs is that the
    comparison should not share a bug with the thing it is checking.
    """
    return {match.group(0) for match in _NUMERAL.finditer(text)}


def test_model_on_and_model_off_produce_the_same_numbers(bundle: EvidenceBundle) -> None:
    with_model = narrate(bundle, ON, client=ReorderingClient(bundle))
    without_model = narrate(bundle, OFF)

    assert with_model.source == "llm"
    assert without_model.source == "template"
    assert with_model.text != without_model.text
    assert numerals(with_model.text) == numerals(without_model.text)


def test_template_output_verifies_against_its_own_bundle(bundle: EvidenceBundle) -> None:
    result = verify_numbers(template_narration(bundle), bundle)
    assert result.ok, f"template emitted unsupported figures: {result.unsupported}"
    assert result.checked > 5


def test_fabricated_figure_is_rejected_and_recorded(bundle: EvidenceBundle) -> None:
    draft = (
        "Fill rate for os_version=Android 15 fell 44.8% over the window. Revenue lost to the "
        "incident was $412,880.19, spread across 37 publishers."
    )
    narration = narrate(bundle, ON, client=FakeClient(draft))

    assert narration.source == "template"
    assert narration.verified is False
    assert narration.text == template_narration(bundle)
    assert "$412,880.19" in narration.unsupported
    assert "37" in narration.unsupported
    # The one figure that was real must not be reported as invented.
    assert not any("44.8" in item for item in narration.unsupported)


def test_faithful_rounding_is_accepted(bundle: EvidenceBundle) -> None:
    draft = (
        "Fill rate for Android 15 fell about 45% over the window, from 78.5% to 43.3%. That "
        "single segment accounts for 93% of the platform-wide movement."
    )
    narration = narrate(bundle, ON, client=FakeClient(draft))

    assert narration.source == "llm", narration.unsupported
    assert narration.verified is True
    assert narration.unsupported == []


@pytest.mark.parametrize(
    ("written", "supported"),
    [
        ("45%", True),
        ("44.8%", True),
        ("44.79%", True),
        ("16%", False),
        ("62%", False),
    ],
)
def test_rounding_tolerance_admits_paraphrase_without_admitting_invention(
    bundle: EvidenceBundle, written: str, supported: bool
) -> None:
    result = verify_numbers(f"The segment fell {written} over the window.", bundle)
    assert result.ok is supported, result.unsupported


@pytest.mark.parametrize(
    "label",
    ["Android 15", "iOS 17.2", "Galaxy S23", "iPhone 14"],
)
def test_digits_inside_segment_names_are_not_figures(bundle: EvidenceBundle, label: str) -> None:
    result = verify_numbers(
        f"The evidence points at {label} rather than at any of its neighbours.", bundle
    )
    assert result.ok, result.unsupported
    assert result.unsupported == []


def test_a_segment_name_never_shown_to_the_model_is_still_not_a_figure(
    bundle: EvidenceBundle,
) -> None:
    """Two separate mechanisms, and both have to work.

    A digit welded to a letter is part of a name whatever the bundle says, so "S24" and "2nd"
    need no vocabulary at all. A digit separated by a space does need one, and the vocabulary
    is the bundle -- which means "Pixel 6", a segment this investigation never looked at, is
    correctly refused rather than waved through. A model naming a segment that is not in the
    evidence is a failure worth catching, not an inconvenience to work around.
    """
    assert verify_numbers("A Galaxy S24 handset behaves the same way.", bundle).ok
    assert verify_numbers("The 2nd half of the window agrees with the 1st.", bundle).ok
    assert not verify_numbers("Pixel 6 was also affected.", bundle).ok


def test_dates_years_and_times_are_not_figures(bundle: EvidenceBundle) -> None:
    prose = (
        "The window opened on 2026-06-23 at 00:00 and closed on 2026-06-26. Traffic recovered "
        "around 14:00 on 23 June, and the same shape appeared in June 2026 a year after the "
        "comparable incident."
    )
    result = verify_numbers(prose, bundle)
    assert result.ok, result.unsupported


def test_a_year_outside_the_investigation_is_still_checked(bundle: EvidenceBundle) -> None:
    """The date exemption is bounded by the window, or it becomes a hole.

    Excusing any four-digit number that looks like a year would let a fabricated count of 1998
    requests through untouched. Only the years the investigation actually spans are exempt.
    """
    assert not verify_numbers("The same pattern was last seen in 1998.", bundle).ok


def test_percent_and_percentage_points_are_not_interchangeable(bundle: EvidenceBundle) -> None:
    """The relative move is 44.79%; the absolute move is 35.16 percentage points.

    Both figures are in the bundle, so a check that only asks "is this number present" passes
    either sentence. Reading the unit is the only thing that separates a correct statement from
    a confident one that is wrong by an order of magnitude.
    """
    assert verify_numbers("Fill rate fell 44.8% for the segment.", bundle).ok
    assert verify_numbers("Fill rate fell 35.16pp for the segment.", bundle).ok
    assert verify_numbers("Fill rate fell 35.2 percentage points.", bundle).ok

    assert not verify_numbers("Fill rate fell 35.2% for the segment.", bundle).ok
    assert not verify_numbers("Fill rate fell 44.8pp for the segment.", bundle).ok
    assert not verify_numbers("Fill rate fell 44.8 percentage points.", bundle).ok


def test_figures_quoted_out_of_a_test_explanation_are_supported(bundle: EvidenceBundle) -> None:
    """The explanations the statistics code writes are evidence too.

    Sufficiency reports "93% of the change is accounted for" and maximality "12% of the 8
    sibling values". Those figures never become claim values, but they are shown to the model,
    printed in the case file, and computed by the same code as everything else. A verifier that
    only knew about claim values would reject the most quotable sentence in the report.
    """
    quoted = (
        "Removing the segment leaves the platform metric almost where it should be: 93% of the "
        "change is accounted for, and only 12% of the 8 sibling values of os_version moved "
        "with it."
    )
    assert verify_numbers(quoted, bundle).ok


def test_evidence_is_identical_whether_or_not_the_model_runs() -> None:
    """The guarantee stated at the level it actually holds: the figures are the artefact.

    Prose is downstream of the bundle and cannot feed back into it, so the deliverable a
    reviewer re-checks is byte-identical with the model on and off. Narrating twice here is the
    point -- if narration could ever mutate the bundle, this is where it would show.
    """
    bundle = build_evidence(_localization(), _finding())
    before = json.dumps(bundle.to_dict(), sort_keys=True)

    narrate(bundle, ON, client=ReorderingClient(bundle))
    narrate(bundle, OFF)
    narrate(bundle, ON, client=FakeClient("Revenue fell by $9,999,999.00 across 61 segments."))

    assert json.dumps(bundle.to_dict(), sort_keys=True) == before


def test_tolerance_bounds_how_far_a_round_number_may_stretch(bundle: EvidenceBundle) -> None:
    """A figure written to the hundred is faithful to the hundred, not to anything at all.

    "275,600" carries no digit below the hundreds, so it cannot be wrong below them; the
    default tolerance caps that licence so a rounder number does not become a wider wildcard.
    Re-rounding is independent of the tolerance, which is why "45%" survives tolerance=0.
    """
    assert verify_numbers("It earned $275,600 in the window.", bundle).ok
    assert not verify_numbers("It earned $275,600 in the window.", bundle, tolerance=0.0).ok
    assert verify_numbers("The segment fell 45% over the window.", bundle, tolerance=0.0).ok


def test_thousands_separators_currency_and_scientific_notation(bundle: EvidenceBundle) -> None:
    assert verify_numbers("The segment carried 20,000 requests.", bundle).ok
    assert verify_numbers("It earned $275,597.12 in the window.", bundle).ok
    assert verify_numbers("The result is significant at p = 1.2e-14.", bundle).ok
    assert not verify_numbers("The segment carried 4,120,000 requests.", bundle).ok


def test_a_client_that_raises_falls_back_without_escaping(bundle: EvidenceBundle) -> None:
    for exc in (TimeoutError("read timed out"), RuntimeError("boom"), ValueError("bad json")):
        narration = narrate(bundle, ON, client=ExplodingClient(exc))
        assert narration.source == "template"
        assert narration.text == template_narration(bundle)


def test_a_failed_or_empty_completion_falls_back(bundle: EvidenceBundle) -> None:
    for client in (
        FakeClient("", ok=False),
        FakeClient("   ", ok=True),
        FakeClient("a draft that never arrived", ok=False),
        UnavailableClient(),
    ):
        narration = narrate(bundle, ON, client=client)
        assert narration.source == "template"
        assert narration.verified is False
        assert narration.text == template_narration(bundle)


def test_disabled_config_never_reaches_the_client(bundle: EvidenceBundle) -> None:
    narration = narrate(bundle, OFF, client=ExplodingClient(AssertionError("must not be called")))
    assert narration.source == "template"
    assert narration.model == ""


def test_prompt_carries_claims_and_no_raw_counters(bundle: EvidenceBundle) -> None:
    prompt = build_prompt(bundle)
    assert "accused.relative_effect" in prompt
    assert "44.8%" in prompt
    assert "-35.16pp" in prompt
    # The model is shown rendered claims, never anything it could divide to make a new figure.
    assert "Counters(" not in prompt


def test_template_reads_as_prose_for_an_accused_case(bundle: EvidenceBundle) -> None:
    text = template_narration(bundle)
    assert "os_version=Android 15" in text
    assert "44.8%" in text
    assert "sufficiency test held" in text
    assert "predicted at" in text
    assert len(text.split()) > 120
    assert "{" not in text and "None" not in text


def test_template_reads_as_prose_for_a_structural_only_case() -> None:
    localization = _localization()
    localization.mode = "structural_only"
    localization.parent_observed = BASE_RATE
    localization.parent_expected = BASE_RATE
    localization.parent_deviation = 0.0
    localization.note = (
        "The parent metric did not move, so there is nothing for a counterfactual to restore. "
        "The candidate is judged on its own deviation and on whether its siblings share it."
    )
    structural = build_evidence(localization, _finding())

    text = template_narration(structural)
    assert "os_version=Android 15" in text
    assert verify_numbers(text, structural).ok
    # The prose must not claim a counterfactual restored a parent that never moved. Describing
    # a test that did not run is the one thing a fallback template must never do.
    assert "stands on its own deviation" in text
    assert "The accusation rests on a counterfactual" not in text


def test_template_reads_as_prose_when_nothing_was_accused() -> None:
    localization = _localization()
    localization.accused = None
    for candidate in localization.candidates:
        candidate.status = "too_broad" if candidate.segment.depth == 1 else "too_narrow"
        candidate.reason = candidate.reason or "Rejected by a breadth test."
    empty = build_evidence(localization, _finding())

    text = template_narration(empty)
    assert "No segment could be held responsible" in text
    assert "rejected for a stated reason" in text
    assert verify_numbers(text, empty).ok

    narration = narrate(empty, OFF)
    assert narration.source == "template"


def test_no_data_mode_says_so_rather_than_guessing() -> None:
    localization = Localization(
        metric="fill_rate",
        window=WINDOW,
        parent=Segment.total(),
        parent_observed=None,
        parent_expected=None,
        parent_deviation=0.0,
        accused=None,
        candidates=[],
        mode="no_data",
        note="No total-level rollup rows exist for this window.",
    )
    empty = build_evidence(localization, _finding())
    text = template_narration(empty)
    assert "no verdict is offered" in text
    assert verify_numbers(text, empty).ok


def test_evidence_bundle_serialises_with_the_stdlib_encoder(bundle: EvidenceBundle) -> None:
    payload = json.dumps(bundle.to_dict())
    restored = json.loads(payload)

    assert restored["metric"] == "fill_rate"
    assert restored["window"]["grain"] == "1d"
    assert {c["key"] for c in restored["claims"]} >= {
        "accused.segment",
        "accused.relative_effect",
        "finding.p_value",
        "checks.run",
    }
    assert any(c["name"] == "sufficiency" for c in restored["checks"])
    assert any(e["segment"] == "device_model=Galaxy S23" for e in restored["cleared"])
    assert "Android 15" in restored["labels"]


def test_non_finite_values_do_not_produce_invalid_json() -> None:
    """An empty denominator can reach the bundle as an infinity.

    ``json.dumps`` writes that as bare ``Infinity``, which no conforming parser will read back,
    so the deliverable fails as a whole rather than in one field.
    """
    localization = _localization()
    localization.parent_expected = float("inf")
    localization.accused.observed_value = float("nan")
    payload = json.dumps(build_evidence(localization, _finding()).to_dict())

    assert "Infinity" not in payload
    assert "NaN" not in payload
    json.loads(payload)


def test_confidence_is_read_defensively_and_included_when_present(bundle: EvidenceBundle) -> None:
    class Component:
        def __init__(self, name, score, weight):
            self.name = name
            self.score = score
            self.weight = weight
            self.state = "pass"
            self.detail = f"{name} scored from a test that was actually run."

    class Score:
        value = 0.82
        components = [Component("sufficiency", 0.93, 0.30), Component("stability", 0.64, 0.15)]
        components_scored = 2
        components_total = 5
        publishable = True
        caveat = "Holdout could not be split, so stability rests on one window."

    scored = build_evidence(_localization(), _finding(), Score())
    keys = {claim.key for claim in scored.claims}
    assert {
        "confidence.value",
        "confidence.sufficiency.score",
        "confidence.components_scored",
        "confidence.publishable",
        "confidence.caveat",
    } <= keys
    # A model writing "2 of the 5 components" must have those digits to resolve against.
    assert verify_numbers("Only 2 of the 5 components could be scored.", scored).ok

    text = template_narration(scored)
    assert "Overall confidence in this verdict is 82.0%" in text
    assert verify_numbers(text, scored).ok
    assert bundle.numbers() != scored.numbers()


def test_a_confidence_object_of_the_wrong_shape_is_ignored_rather_than_fatal() -> None:
    class Broken:
        value = "not a number"
        components = "not a list"

    scored = build_evidence(_localization(), _finding(), Broken())
    assert not any(claim.key.startswith("confidence.") for claim in scored.claims)
    assert template_narration(scored)


def test_client_is_unavailable_without_a_key_and_never_raises() -> None:
    client = LLMClient(LLMConfig(enabled=False))
    assert client.available is False

    completion = client.complete("system", "user")
    assert completion.ok is False
    assert completion.text == ""
    assert "disabled" in completion.error.lower()


def test_client_construction_is_offline_and_honours_base_url() -> None:
    """Constructing a client must not open a connection, or importing the CLI would hang."""
    client = LLMClient(
        LLMConfig(enabled=True, api_key="not-a-real-key", base_url="http://127.0.0.1:9/v1")
    )
    assert client.available is True
    assert client.error == ""


def test_verification_counts_what_it_checked(bundle: EvidenceBundle) -> None:
    result = verify_numbers("Nothing numeric here at all.", bundle)
    assert result.ok is True
    assert result.checked == 0

    result = verify_numbers("It fell 44.8% and 12.0% and 91.4%.", bundle)
    assert result.checked == 3
    assert result.unsupported == ["91.4%"]
