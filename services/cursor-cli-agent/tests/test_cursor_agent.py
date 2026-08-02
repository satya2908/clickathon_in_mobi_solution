import json
import stat
from pathlib import Path

import pytest

from app.config import AgentConfig
from app.cursor_agent import (
    AgentUnavailableError,
    ContextPathError,
    CursorAgentClient,
)


def fake_agent(tmp_path: Path) -> Path:
    script = tmp_path / "agent"
    script.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

assert os.environ["CURSOR_API_KEY"] == "test-secret"
assert "--api-key" not in sys.argv
assert sys.argv[sys.argv.index("--agent-endpoint") + 1] == "https://api2.cursor.sh"
assert "CURSOR_SERVICE_API_TOKEN" not in os.environ
chats = Path(os.environ["HOME"], ".cursor", "chats")
chats.mkdir(parents=True, exist_ok=True)
Path(chats, "child-marker").write_text("created")
model = sys.argv[sys.argv.index("--model") + 1]
print(json.dumps({
    "result": "completed",
    "session_id": "session-123",
    "request_id": "request-123",
    "duration_ms": 12,
    "usage": {"input_tokens": 10},
    "model": model,
}))
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def make_client(tmp_path: Path, *, with_mcp: bool = False) -> CursorAgentClient:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "case.json").write_text('{"case": 1}\n', encoding="utf-8")
    config = AgentConfig(
        binary=str(fake_agent(tmp_path)),
        agent_endpoint="https://api2.cursor.sh",
        timeout_seconds=5,
        workspace_root=workspace,
        runtime_dir=tmp_path / "runtime",
        cursor_home=tmp_path / "home",
        **(
            {
                "mcp_config": {
                    "mcpServers": {
                        "verdict-clickhouse": {"url": "http://mcp-clickhouse:8000/sse"}
                    }
                },
                "mcp_allowed_servers": ["verdict-clickhouse"],
            }
            if with_mcp
            else {}
        ),
    )
    client = CursorAgentClient(config, "test-secret")
    client.prepare()
    return client


def test_runs_agent_with_env_auth_and_parses_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURSOR_SERVICE_API_TOKEN", "must-not-reach-child")
    client = make_client(tmp_path)

    result = client.run(
        "Review this case",
        model="requested-model",
        context={"metric": "fill_rate"},
        context_files=["case.json"],
    )

    assert result.output == "completed"
    assert result.session_id == "session-123"
    assert result.model == "requested-model"
    assert result.usage == {"input_tokens": 10}

    policy = json.loads((client.config.cursor_home / ".cursor" / "cli-config.json").read_text())
    assert policy["network"]["useHttp1ForAgent"] is True
    assert "Write(**)" in policy["permissions"]["deny"]
    assert "Read(**/.env*)" in policy["permissions"]["deny"]
    assert "Mcp(*:*)" in policy["permissions"]["deny"]
    assert (client.config.cursor_home / "_shared" / "chats" / "child-marker").exists()


def test_writes_only_explicitly_allowed_mcp_servers(tmp_path: Path) -> None:
    client = make_client(tmp_path, with_mcp=True)

    policy = json.loads((client.config.cursor_home / ".cursor" / "cli-config.json").read_text())
    mcp = json.loads((client.config.cursor_home / ".cursor" / "mcp.json").read_text())

    assert "Mcp(verdict-clickhouse:*)" in policy["permissions"]["allow"]
    assert "Mcp(*:*)" not in policy["permissions"]["deny"]
    assert mcp == {
        "mcpServers": {"verdict-clickhouse": {"url": "http://mcp-clickhouse:8000/sse"}}
    }


def test_ephemeral_home_does_not_persist_remediation_session(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    client.run(
        "Generate advice",
        model="requested-model",
        ephemeral_home=True,
    )

    assert not (client.config.cursor_home / "_shared" / "chats" / "child-marker").exists()


def test_prepare_rejects_symlinked_request_spool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (runtime / "requests").symlink_to(unrelated, target_is_directory=True)
    client = CursorAgentClient(
        AgentConfig(
            binary=str(fake_agent(tmp_path)),
            workspace_root=workspace,
            runtime_dir=runtime,
            cursor_home=tmp_path / "home",
        ),
        "test-secret",
    )

    with pytest.raises(AgentUnavailableError, match="must not be a symlink"):
        client.prepare()

    assert sentinel.exists()


def test_rejects_context_outside_workspace(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("not approved", encoding="utf-8")

    with pytest.raises(ContextPathError, match="must be inside"):
        client.run(
            "Review",
            model="requested-model",
            context_files=[str(outside)],
        )
