"""The model client, written so that a narration failure cannot take an investigation down.

Every path out of here returns a ``Completion``. Nothing raises. A disabled config, an absent
key, a refused connection, a read timeout, a 500 from the endpoint, or a response shaped
differently by a provider that is only approximately OpenAI-compatible all arrive at the caller
as ``ok=False`` carrying the reason, and the caller falls back to template prose. Letting an
exception escape would mean a transient network fault destroys a case file whose figures were
already computed, verified and correct: the analysis would have been right, and the run would
still have died on the way to describing it.

``base_url`` decides where the request goes, so a self-hosted vLLM, a gateway, or a different
vendor works unchanged. Nothing here assumes OpenAI the company, only the wire format.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import LLMConfig

log = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


@dataclass(frozen=True)
class Completion:
    """One model response, or the reason there is not one.

    The usage counters are recorded even on the failure paths, because the interesting question
    about a discarded draft is usually what it cost and how long it took, and a call that was
    refused after two retries is not free.
    """

    text: str
    model: str
    ok: bool
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


def _usage(response: Any) -> tuple[int, int]:
    """Prompt and completion token counts, zero where the provider does not report them.

    Read defensively: ``usage`` is optional in the wire format and several compatible servers
    omit it entirely. Note that on reasoning models ``completion_tokens`` excludes the hidden
    thinking, so it will not reconcile with ``total_tokens`` -- the gap between them is the
    thinking, and it is the quantity that explains a truncated draft.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    def count(name: str) -> int:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) and value >= 0 else 0

    return count("prompt_tokens"), count("completion_tokens")


def _extract_text(response: Any) -> str:
    """Pull the assistant message out of a response without trusting its shape.

    Gateways and self-hosted servers claiming OpenAI compatibility return empty ``choices``,
    omit ``message``, or set ``content`` to null on a refusal or a tool call. Indexing straight
    into the structure turns any of those into an ``IndexError`` or ``AttributeError`` several
    frames away from the cause, which then reads like a bug in the narration layer.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _stopped_early(response: Any) -> bool:
    """Whether the model ran out of budget rather than finishing.

    Read defensively for the same reason as ``_extract_text``: ``finish_reason`` is absent on
    some compatible servers, and a missing field must mean "no evidence of truncation" rather
    than raising. Only the explicit ``length`` signal counts.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    return getattr(choices[0], "finish_reason", None) == "length"


class LLMClient:
    """A thin wrapper over an OpenAI-compatible endpoint whose contract is that it never raises."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._error = ""

        if not cfg.enabled:
            self._error = "Narration is disabled in config."
            return
        if not cfg.api_key:
            self._error = "No API key is configured for the narration endpoint."
            return

        # Imported here rather than at module scope so that a deployment which never narrates
        # does not need the SDK installed at all, and so that a broken install -- a partial
        # wheel, an httpx major version the SDK will not accept -- degrades to template prose
        # instead of breaking every import of the package, including the CLI that would have
        # been used to diagnose it.
        try:
            from openai import OpenAI
        except Exception as exc:
            self._error = f"OpenAI SDK unavailable: {type(exc).__name__}: {exc}"
            return

        try:
            self._client = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                timeout=float(cfg.timeout_seconds),
                # The SDK's own retry loop, so a 429 or a dropped connection is retried with
                # backoff rather than immediately costing the case file its prose. Zero is a
                # legal setting and means a single attempt.
                max_retries=cfg.max_retries,
            )
        except Exception as exc:
            self._error = (
                f"Could not construct a client for {cfg.base_url}: {type(exc).__name__}: {exc}"
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def error(self) -> str:
        """Why the client is unavailable, recorded so a silent fallback is still explicable."""
        return self._error

    def complete(self, system: str, user: str) -> Completion:
        if self._client is None:
            return Completion("", self.cfg.model, False, self._error)

        extra: dict[str, Any] = {}
        if self.cfg.reasoning_effort:
            extra["reasoning_effort"] = self.cfg.reasoning_effort

        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        except Exception as exc:
            # Deliberately every exception, not a list of the SDK's own error classes. The
            # failures worth surviving here include ones the SDK does not own: a proxy closing
            # the socket, a DNS failure, a JSON body the client cannot parse. Enumerating
            # classes would mean the unanticipated failure is the one that escapes.
            log.warning("Narration call to %s failed: %s", self.cfg.base_url, exc)
            return Completion(
                "", self.cfg.model, False, f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(started),
            )

        elapsed = _elapsed_ms(started)
        prompt_tokens, completion_tokens = _usage(response)

        text = _extract_text(response).strip()
        if not text:
            return Completion(
                "", self.cfg.model, False, "The endpoint returned no content.",
                prompt_tokens, completion_tokens, elapsed,
            )

        # A response cut off at the token ceiling is rejected rather than published. The numeric
        # verifier downstream cannot catch this: it checks that every figure was computed, and a
        # sentence ending "This segment" contains no wrong figure, so a half-written paragraph
        # passes it cleanly and reaches the case file. Worse, a cut landing inside a number turns
        # "-44.8%" into a bare "6", which then reads as an invented figure and gets blamed on the
        # model. Both were observed against Gemini Flash, whose thinking tokens are charged
        # against max_tokens without appearing in the text.
        if _stopped_early(response):
            return Completion(
                "", self.cfg.model, False,
                f"The response hit the {self.cfg.max_tokens}-token ceiling before finishing. "
                "Raise llm.max_tokens, or set llm.reasoning_effort low if the model bills "
                "hidden reasoning against that budget.",
                prompt_tokens, completion_tokens, elapsed,
            )

        # The served model can differ from the one requested -- an alias resolving to a dated
        # snapshot, a gateway routing to a fallback -- and the case file should record what
        # actually wrote the prose rather than what was asked for.
        served = getattr(response, "model", None)
        return Completion(
            text, served if isinstance(served, str) and served else self.cfg.model, True, "",
            prompt_tokens, completion_tokens, elapsed,
        )
