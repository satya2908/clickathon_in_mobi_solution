import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import (
    AgentConfig,
    LoadedConfig,
    ModelConfig,
    RemediationConfig,
    ServerConfig,
    ServiceConfig,
    StorageConfig,
)
from app.cursor_agent import AgentResult
from app.main import create_app


def recommendation_json(title: str) -> str:
    return json.dumps(
        {
            "summary": "reviewed",
            "recommendations": [
                {
                    "title": title,
                    "action": "Canary the targeted change.",
                    "rationale": "The supplied case supports it.",
                    "expected_benefit": "Reduces recurrence risk.",
                    "validation_step": "Compare the canary to a control.",
                    "risk": "Rollback if the target metric falls.",
                    "priority": "high",
                    "confidence": "medium",
                    "evidence": ["case evidence"],
                }
            ],
        }
    )


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def prepare(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def ready(self) -> tuple[bool, str]:
        return True, ""

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        with self.lock:
            self.calls.append({"prompt": prompt, **kwargs})
            output = self.outputs.pop(0)
        return AgentResult(
            output=output,
            model=kwargs["model"],
            session_id="session-1",
            request_id="request-1",
            duration_ms=5,
        )


def loaded_config(tmp_path: Path) -> LoadedConfig:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = ServiceConfig(
        server=ServerConfig(max_request_bytes=1024),
        models=ModelConfig(
            default="default-model",
            remediation="generation-model",
            validation="validation-model",
        ),
        agent=AgentConfig(
            workspace_root=workspace,
            runtime_dir=tmp_path / "runtime",
            cursor_home=tmp_path / "home",
        ),
        remediation=RemediationConfig(max_recommendations=3, max_pending_jobs=1),
        storage=StorageConfig(jobs_dir=tmp_path / "jobs"),
    )
    return LoadedConfig(
        settings=settings,
        cursor_api_key="cursor-secret",
        service_api_token="service-secret",
        path=tmp_path / "config.yaml",
    )


def test_run_uses_configured_model_and_bearer_auth(tmp_path: Path) -> None:
    fake = FakeClient(["answer"])
    api = create_app(loaded_config(tmp_path), client=fake)  # type: ignore[arg-type]

    with TestClient(api) as client:
        unauthorized = client.post("/v1/run", json={"prompt": "hello"})
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer service-secret"},
            json={"prompt": "hello"},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["model"] == "default-model"
    assert fake.calls[0]["model"] == "default-model"


def test_rejects_oversized_request_before_agent_call(tmp_path: Path) -> None:
    fake = FakeClient(["unused"])
    api = create_app(loaded_config(tmp_path), client=fake)  # type: ignore[arg-type]

    with TestClient(api) as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer service-secret"},
            json={"prompt": "x" * 2_000},
        )

    assert response.status_code == 413
    assert fake.calls == []


def test_rejects_interactive_request_when_waiter_limit_is_full(tmp_path: Path) -> None:
    fake = FakeClient(["unused"])
    api = create_app(loaded_config(tmp_path), client=fake)  # type: ignore[arg-type]

    with TestClient(api) as client:
        api.state.interactive_requests = (
            api.state.loaded_config.settings.agent.max_interactive_requests
        )
        response = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer service-secret"},
            json={"message": "hello"},
        )
        api.state.interactive_requests = 0

    assert response.status_code == 429
    assert fake.calls == []


def test_rejects_remediation_when_bounded_queue_is_full(tmp_path: Path) -> None:
    fake = FakeClient([recommendation_json("unused")])
    api = create_app(loaded_config(tmp_path), client=fake)  # type: ignore[arg-type]

    with TestClient(api) as client:
        api.state.background_tasks["synthetic-busy-job"] = object()
        response = client.post(
            "/v1/remediations",
            headers={"Authorization": "Bearer service-secret"},
            json={"case_id": "case-2", "case_data": {"metric": "ctr"}},
        )
        api.state.background_tasks.pop("synthetic-busy-job")

    assert response.status_code == 429
    assert fake.calls == []


def test_remediation_runs_generate_then_independent_validate(tmp_path: Path) -> None:
    fake = FakeClient(
        [
            recommendation_json("draft"),
            recommendation_json("validated"),
        ]
    )
    api = create_app(loaded_config(tmp_path), client=fake)  # type: ignore[arg-type]
    headers = {"Authorization": "Bearer service-secret"}

    with TestClient(api) as client:
        accepted = client.post(
            "/v1/remediations",
            headers=headers,
            json={
                "case_id": "case-1",
                "case_data": {"metric": "fill_rate", "segment": "os=15"},
            },
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]

        job = None
        for _ in range(100):
            response = client.get(
                f"/v1/remediations/{job_id}",
                headers=headers,
            )
            job = response.json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["recommendations"][0]["title"] == "validated"
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == "generation-model"
    assert fake.calls[1]["model"] == "validation-model"
    assert fake.calls[0].get("session_id") is None
    assert fake.calls[1].get("session_id") is None
    assert fake.calls[0]["ephemeral_home"] is True
    assert fake.calls[1]["ephemeral_home"] is True
