"""Validated API and persistence models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,127}$"
SESSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,255}$"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=2_000_000)
    model: str | None = Field(default=None, pattern=MODEL_PATTERN)
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN)
    context: Any | None = None
    context_files: list[str] = Field(default_factory=list, max_length=25)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot contain only whitespace")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2_000_000)
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN)
    model: str | None = Field(default=None, pattern=MODEL_PATTERN)
    context: Any | None = None
    context_files: list[str] = Field(default_factory=list, max_length=25)

    @field_validator("message")
    @classmethod
    def message_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message cannot contain only whitespace")
        return value


class AgentResponse(BaseModel):
    success: bool = True
    output: str
    session_id: str | None = None
    request_id: str | None = None
    model: str
    duration_ms: int
    usage: dict[str, Any] = Field(default_factory=dict)


class RemediationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=256)
    case_data: dict[str, Any]
    additional_context: str = Field(default="", max_length=100_000)
    context_files: list[str] = Field(default_factory=list, max_length=25)
    generation_model: str | None = Field(default=None, pattern=MODEL_PATTERN)
    validation_model: str | None = Field(default=None, pattern=MODEL_PATTERN)
    max_recommendations: int | None = Field(default=None, ge=1, le=10)

    @field_validator("case_id")
    @classmethod
    def case_id_must_not_be_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("case_id cannot contain only whitespace")
        return value


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=160)
    action: str = Field(min_length=1, max_length=800)
    rationale: str = Field(min_length=1, max_length=800)
    expected_benefit: str = Field(default="", max_length=500)
    validation_step: str = Field(min_length=1, max_length=800)
    risk: str = Field(default="", max_length=500)
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence: list[str] = Field(default_factory=list, max_length=8)


class RecommendationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(default="", max_length=1_000)
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def utc_now() -> datetime:
    return datetime.now(UTC)


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    case_id: str
    status: JobStatus
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    generation_model: str
    validation_model: str
    draft_recommendations: int = 0
    result: RecommendationResult | None = None
    error: str = ""


class JobAccepted(BaseModel):
    job_id: str
    case_id: str
    status: JobStatus
    status_url: str


class ServiceError(BaseModel):
    error: str
    detail: str = ""
