import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.cursor_agent import AgentResult
from app.jobs import JobStore
from app.models import JobStatus
from app.service import AgentRuntime, extract_json_object, parse_recommendations

VALID_RESULT = """
{
  "summary": "Validated.",
  "recommendations": [
    {
      "title": "Guard the rollout",
      "action": "Canary the configuration change.",
      "rationale": "The case isolates the affected segment.",
      "expected_benefit": "Limits exposure.",
      "validation_step": "Compare the canary with its control.",
      "risk": "Rollback if the metric worsens.",
      "priority": "high",
      "confidence": "high",
      "evidence": ["localized segment"]
    }
  ]
}
"""


class ConcurrentFakeClient:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def run(self, _prompt: str, **kwargs: object) -> AgentResult:
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return AgentResult(
            output="done",
            model=str(kwargs["model"]),
            session_id=None,
            request_id=None,
            duration_ms=1,
        )


def test_extracts_fenced_json() -> None:
    parsed = extract_json_object(f"```json\n{VALID_RESULT}\n```")
    assert parsed["summary"] == "Validated."


def test_parses_validated_recommendations() -> None:
    parsed = parse_recommendations(VALID_RESULT, maximum=5)
    assert len(parsed.recommendations) == 1
    assert parsed.recommendations[0].priority == "high"


def test_different_chat_sessions_execute_concurrently() -> None:
    fake = ConcurrentFakeClient()

    async def exercise() -> None:
        runtime = AgentRuntime(fake, max_concurrency=2)  # type: ignore[arg-type]
        await asyncio.gather(
            runtime.run("one", model="model", session_id="session-one"),
            runtime.run("two", model="model", session_id="session-two"),
        )

    asyncio.run(exercise())
    assert fake.maximum_active == 2


def test_same_chat_session_is_serialized() -> None:
    fake = ConcurrentFakeClient()

    async def exercise() -> None:
        runtime = AgentRuntime(fake, max_concurrency=2)  # type: ignore[arg-type]
        await asyncio.gather(
            runtime.run("one", model="model", session_id="same-session"),
            runtime.run("two", model="model", session_id="same-session"),
        )

    asyncio.run(exercise())
    assert fake.maximum_active == 1


def test_interrupted_jobs_are_marked_failed_on_restart(tmp_path: Path) -> None:
    first = JobStore(tmp_path)
    first.initialize()
    created = first.create(
        case_id="case-1",
        generation_model="model-a",
        validation_model="model-b",
    )
    first.update(created.job_id, status=JobStatus.RUNNING)

    restarted = JobStore(tmp_path)
    restarted.initialize()
    recovered = restarted.get(created.job_id)

    assert recovered is not None
    assert recovered.status == JobStatus.FAILED
    assert "restarted" in recovered.error


def test_terminal_job_history_is_bounded(tmp_path: Path) -> None:
    store = JobStore(tmp_path, max_records=1)
    store.initialize()
    first = store.create(
        case_id="case-1",
        generation_model="model-a",
        validation_model="model-b",
    )
    store.update(first.job_id, status=JobStatus.COMPLETED)
    second = store.create(
        case_id="case-2",
        generation_model="model-a",
        validation_model="model-b",
    )
    store.update(second.job_id, status=JobStatus.COMPLETED)

    assert store.get(first.job_id) is None
    assert store.get(second.job_id) is not None


def test_get_prunes_records_that_expire_while_service_is_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path, retention_days=1)
    store.initialize()
    job = store.create(
        case_id="case-1",
        generation_model="model-a",
        validation_model="model-b",
    )
    store.update(job.job_id, status=JobStatus.COMPLETED)
    future = datetime.now(UTC) + timedelta(days=2)
    monkeypatch.setattr("app.jobs.utc_now", lambda: future)

    assert store.get(job.job_id) is None
