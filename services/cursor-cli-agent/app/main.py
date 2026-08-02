"""FastAPI surface for one-shot, chat, and two-pass remediation calls."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from . import __version__
from .config import LoadedConfig, load_config
from .cursor_agent import (
    AgentError,
    AgentExecutionError,
    AgentInputTooLargeError,
    AgentTimeoutError,
    AgentUnavailableError,
    ContextPathError,
    CursorAgentClient,
)
from .jobs import JobConflictError, JobStore
from .middleware import RequestSizeLimitMiddleware
from .models import (
    AgentResponse,
    ChatRequest,
    JobAccepted,
    JobRecord,
    RemediationRequest,
    RunRequest,
)
from .service import AgentRuntime, RemediationService

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def create_app(
    loaded: LoadedConfig | None = None,
    *,
    client: CursorAgentClient | None = None,
) -> FastAPI:
    loaded = loaded or load_config()
    settings = loaded.settings
    cursor_client = client or CursorAgentClient(
        settings.agent,
        loaded.cursor_api_key,
    )
    runtime = AgentRuntime(cursor_client, settings.agent.max_concurrency)
    jobs = JobStore(
        settings.storage.jobs_dir,
        retention_days=settings.storage.retention_days,
        max_records=settings.storage.max_records,
    )
    remediations = RemediationService(
        runtime,
        jobs,
        default_max_recommendations=settings.remediation.max_recommendations,
        max_case_bytes=settings.remediation.max_case_bytes,
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        cursor_client.prepare()
        jobs.initialize()
        api.state.background_tasks = {}
        api.state.interactive_requests = 0
        log.info(
            "Cursor CLI service ready; default=%s remediation=%s validation=%s",
            settings.models.default,
            settings.models.remediation,
            settings.models.validation,
        )
        try:
            yield
        finally:
            cursor_client.shutdown()
            tasks: list[asyncio.Task[Any]] = list(api.state.background_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    api = FastAPI(
        title="Cursor CLI Agent",
        description="Thin, read-only wrapper for Cursor Agent and validated remediations.",
        version=__version__,
        lifespan=lifespan,
    )
    api.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=settings.server.max_request_bytes,
    )
    api.state.loaded_config = loaded
    api.state.cursor_client = cursor_client
    api.state.runtime = runtime
    api.state.jobs = jobs
    api.state.remediations = remediations

    async def require_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = loaded.service_api_token
        if not expected:
            return
        prefix = "Bearer "
        supplied = (
            authorization[len(prefix) :]
            if authorization and authorization.startswith(prefix)
            else ""
        )
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    protected = Depends(require_token)

    @api.exception_handler(AgentError)
    async def agent_error_handler(
        _request: Request,
        exc: AgentError,
    ) -> JSONResponse:
        if isinstance(exc, ContextPathError):
            code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, AgentInputTooLargeError):
            code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        elif isinstance(exc, AgentUnavailableError):
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, AgentTimeoutError):
            code = status.HTTP_504_GATEWAY_TIMEOUT
        elif isinstance(exc, AgentExecutionError):
            code = status.HTTP_502_BAD_GATEWAY
        else:
            code = status.HTTP_502_BAD_GATEWAY
        return JSONResponse(
            status_code=code,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @api.get("/")
    async def root() -> dict[str, str]:
        return {"service": "cursor-cli-agent", "version": __version__}

    @api.get("/health")
    async def health(response: Response) -> dict[str, Any]:
        ready, reason = cursor_client.ready()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "ok" if ready else "not_ready",
            "reason": reason,
            "binary": settings.agent.binary,
            "models": {
                "default": settings.models.default,
                "remediation": settings.models.remediation,
                "validation": settings.models.validation,
            },
            "authentication_enabled": bool(loaded.service_api_token),
        }

    @api.get("/v1/config", dependencies=[protected])
    async def public_config() -> dict[str, Any]:
        return settings.public_dict()

    def resolve_model(
        purpose: str,
        requested: str | None,
    ) -> str:
        try:
            return settings.models.resolve(purpose, requested)  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    async def run_interactive(*args: Any, **kwargs: Any) -> Any:
        if api.state.interactive_requests >= settings.agent.max_interactive_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="interactive agent queue is full; retry later",
            )
        api.state.interactive_requests += 1
        try:
            return await runtime.run(*args, **kwargs)
        finally:
            api.state.interactive_requests -= 1

    @api.post(
        "/v1/run",
        response_model=AgentResponse,
        dependencies=[protected],
    )
    async def run_agent(request: RunRequest) -> AgentResponse:
        model = resolve_model("default", request.model)
        result = await run_interactive(
            request.prompt,
            model=model,
            session_id=request.session_id,
            context=request.context,
            context_files=request.context_files,
        )
        return AgentResponse(
            output=result.output,
            session_id=result.session_id,
            request_id=result.request_id,
            model=result.model,
            duration_ms=result.duration_ms,
            usage=result.usage,
        )

    @api.post(
        "/v1/chat",
        response_model=AgentResponse,
        dependencies=[protected],
    )
    async def chat(request: ChatRequest) -> AgentResponse:
        model = resolve_model("default", request.model)
        result = await run_interactive(
            request.message,
            model=model,
            session_id=request.session_id,
            context=request.context,
            context_files=request.context_files,
        )
        return AgentResponse(
            output=result.output,
            session_id=result.session_id,
            request_id=result.request_id,
            model=result.model,
            duration_ms=result.duration_ms,
            usage=result.usage,
        )

    @api.post(
        "/v1/remediations",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[protected],
    )
    async def create_remediation(request: RemediationRequest) -> JobAccepted:
        if len(api.state.background_tasks) >= settings.remediation.max_pending_jobs:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "remediation queue is full; retry after an existing job reaches "
                    "a terminal state"
                ),
            )
        try:
            remediations.validate_case_size(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(exc),
            ) from exc

        generation_model = resolve_model(
            "remediation",
            request.generation_model,
        )
        validation_model = resolve_model(
            "validation",
            request.validation_model,
        )
        try:
            job = jobs.create(
                case_id=request.case_id,
                generation_model=generation_model,
                validation_model=validation_model,
            )
        except JobConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": str(exc),
                    "job_id": exc.job.job_id,
                    "status": exc.job.status,
                },
            ) from exc

        task = asyncio.create_task(
            remediations.run_job(job, request),
            name=f"remediation-{job.job_id}",
        )
        api.state.background_tasks[job.job_id] = task

        def discard(_task: asyncio.Task[Any]) -> None:
            api.state.background_tasks.pop(job.job_id, None)
            if not _task.cancelled():
                exception = _task.exception()
                if exception is not None:
                    log.error(
                        "Remediation task %s escaped its failure handler",
                        job.job_id,
                        exc_info=(
                            type(exception),
                            exception,
                            exception.__traceback__,
                        ),
                    )

        task.add_done_callback(discard)
        return JobAccepted(
            job_id=job.job_id,
            case_id=job.case_id,
            status=job.status,
            status_url=f"/v1/remediations/{job.job_id}",
        )

    @api.get(
        "/v1/remediations/{job_id}",
        response_model=JobRecord,
        dependencies=[protected],
    )
    async def get_remediation(job_id: str) -> JobRecord:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="remediation job not found",
            )
        return job

    return api


app = create_app()
