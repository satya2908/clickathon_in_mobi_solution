"""Concurrency control and the two-pass remediation workflow."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError

from .cursor_agent import AgentResult, CursorAgentClient
from .jobs import JobStore
from .models import (
    JobRecord,
    JobStatus,
    RecommendationResult,
    RemediationRequest,
)
from .prompts import generation_prompt, validation_prompt

log = logging.getLogger(__name__)


class RecommendationOutputError(RuntimeError):
    """The validator response did not satisfy the recommendation contract."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from plain text or a fenced model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RecommendationOutputError("response did not contain a JSON object")


def parse_recommendations(text: str, maximum: int) -> RecommendationResult:
    try:
        result = RecommendationResult.model_validate(extract_json_object(text))
    except (ValidationError, RecommendationOutputError) as exc:
        raise RecommendationOutputError(
            f"validator response did not match the recommendation schema: {exc}"
        ) from exc
    if len(result.recommendations) > maximum:
        result.recommendations = result.recommendations[:maximum]
    return result


class AgentRuntime:
    def __init__(self, client: CursorAgentClient, max_concurrency: int) -> None:
        self.client = client
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._session_locks = tuple(asyncio.Lock() for _ in range(128))

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[None]:
        async with self._semaphore:
            yield

    async def run_reserved(self, *args: Any, **kwargs: Any) -> AgentResult:
        return await asyncio.to_thread(self.client.run, *args, **kwargs)

    async def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        session_id = kwargs.get("session_id")
        if isinstance(session_id, str) and session_id:
            lock = self._session_locks[hash(session_id) % len(self._session_locks)]
            async with lock:
                async with self.reserve():
                    return await self.run_reserved(*args, **kwargs)
        async with self.reserve():
            return await self.run_reserved(*args, **kwargs)


class RemediationService:
    def __init__(
        self,
        runtime: AgentRuntime,
        jobs: JobStore,
        *,
        default_max_recommendations: int,
        max_case_bytes: int,
    ) -> None:
        self.runtime = runtime
        self.jobs = jobs
        self.default_max_recommendations = default_max_recommendations
        self.max_case_bytes = max_case_bytes

    def validate_case_size(self, request: RemediationRequest) -> None:
        serialized = json.dumps(
            request.case_data,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        size = len(serialized.encode("utf-8"))
        if size > self.max_case_bytes:
            raise ValueError(f"case_data is {size} bytes; limit is {self.max_case_bytes}")

    async def run_job(
        self,
        job: JobRecord,
        request: RemediationRequest,
    ) -> None:
        maximum = request.max_recommendations or self.default_max_recommendations
        try:
            async with self.runtime.reserve():
                self.jobs.update(job.job_id, status=JobStatus.RUNNING, error="")
                first_context = {
                    "case_id": request.case_id,
                    "case": request.case_data,
                    "additional_context": request.additional_context,
                }
                generated = await self.runtime.run_reserved(
                    generation_prompt(maximum),
                    model=job.generation_model,
                    context=first_context,
                    context_files=request.context_files,
                    ephemeral_home=True,
                )

                draft_count = 0
                try:
                    draft = RecommendationResult.model_validate(
                        extract_json_object(generated.output)
                    )
                    draft_count = len(draft.recommendations)
                except (ValidationError, RecommendationOutputError):
                    pass

                second_context = {
                    "case_id": request.case_id,
                    "case": request.case_data,
                    "additional_context": request.additional_context,
                    "first_pass_draft": generated.output,
                }
                validated = await self.runtime.run_reserved(
                    validation_prompt(maximum),
                    model=job.validation_model,
                    context=second_context,
                    context_files=request.context_files,
                    ephemeral_home=True,
                )
                result = parse_recommendations(validated.output, maximum)
                self.jobs.update(
                    job.job_id,
                    status=JobStatus.COMPLETED,
                    result=result,
                    draft_recommendations=draft_count,
                    error="",
                )
        except asyncio.CancelledError:
            self.jobs.update(
                job.job_id,
                status=JobStatus.CANCELLED,
                error="job cancelled during service shutdown",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - every job needs a terminal state
            log.warning("Remediation job %s failed: %s", job.job_id, exc)
            self.jobs.update(
                job.job_id,
                status=JobStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}"[:2_000],
            )
