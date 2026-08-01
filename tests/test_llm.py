"""The narration client, whose contract is that it never raises and never half-speaks.

Narration is the last step before a finished case file is returned, so every failure here has to
degrade to a template rather than take an investigation down with it. These tests drive a stub in
place of the SDK, because what is being checked is how the client reacts to a response shape, not
whether any particular vendor produces it.
"""

from __future__ import annotations

from types import SimpleNamespace

from verdict.config import LLMConfig
from verdict.llm import LLMClient


class TestATruncatedDraftIsRejected:
    """A response cut off at the token ceiling must not reach the case file.

    The numeric verifier downstream cannot catch this. It checks that every figure was computed,
    and a paragraph ending "This segment" contains no wrong figure, so a half-written draft
    passes cleanly. Observed against Gemini Flash, which charges invisible reasoning tokens
    against max_tokens: the prose stopped mid-sentence, and where the cut landed inside a number
    it left a bare "6" that then looked like the model had invented a figure.
    """

    def _response(self, text: str, finish_reason: str):
        message = SimpleNamespace(content=text)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], model="gemini-flash-latest")

    def _client(self, response):
        cfg = LLMConfig(enabled=True, api_key="k", max_tokens=3000)
        client = LLMClient(cfg)
        client._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            )
        )
        return client

    def test_a_length_stop_is_not_published(self):
        client = self._client(self._response("The segment carried 54,225 requests and", "length"))
        out = client.complete("s", "u")
        assert not out.ok
        assert not out.text

    def test_the_error_names_the_remedy(self):
        client = self._client(self._response("cut off here", "length"))
        assert "reasoning_effort" in client.complete("s", "u").error

    def test_a_normal_stop_is_published(self):
        client = self._client(self._response("A complete sentence.", "stop"))
        out = client.complete("s", "u")
        assert out.ok
        assert out.text == "A complete sentence."

    def test_a_missing_finish_reason_is_not_treated_as_truncation(self):
        """Some compatible servers omit the field; absence is not evidence of a cut."""
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Fine."))],
            model="whatever",
        )
        assert self._client(response).complete("s", "u").ok


class TestReasoningEffortIsOnlySentWhenSet:
    """Providers disagree: Gemini takes "low" and rejects "none" with a 400, and an endpoint that
    has never heard of the parameter fails the whole request rather than ignoring it."""

    def _capture(self, cfg):
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"), finish_reason="stop"
                    )
                ],
                model="m",
            )

        client = LLMClient(cfg)
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        client.complete("s", "u")
        return seen

    def test_omitted_when_blank(self):
        seen = self._capture(LLMConfig(enabled=True, api_key="k"))
        assert "reasoning_effort" not in seen

    def test_forwarded_when_configured(self):
        seen = self._capture(LLMConfig(enabled=True, api_key="k", reasoning_effort="low"))
        assert seen["reasoning_effort"] == "low"
