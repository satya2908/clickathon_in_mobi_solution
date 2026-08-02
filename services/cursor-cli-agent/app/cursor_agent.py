"""Minimal, non-streaming subprocess wrapper for the Cursor Agent CLI."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AgentConfig


class AgentError(RuntimeError):
    """Base class for expected Cursor Agent failures."""


class AgentUnavailableError(AgentError):
    """The binary or API key is unavailable."""


class AgentTimeoutError(AgentError):
    """The Cursor Agent process exceeded its configured deadline."""


class AgentExecutionError(AgentError):
    """The Cursor Agent process returned a failure."""


class AgentOutputError(AgentError):
    """The process completed but did not return usable JSON."""


class AgentInputTooLargeError(AgentError):
    """A prompt or context exceeds the configured byte limit."""


class ContextPathError(AgentError):
    """A requested context path is outside the allowed workspace."""


@dataclass(frozen=True)
class AgentResult:
    output: str
    model: str
    session_id: str | None
    request_id: str | None
    duration_ms: int
    usage: dict[str, Any] = field(default_factory=dict)


class CursorAgentClient:
    """Run one headless Cursor Agent invocation at a time per call.

    MCP access is restricted to the configured server allow-list. Authentication is passed
    through ``CURSOR_API_KEY`` in the child environment so it never appears in the process
    argument list.
    """

    def __init__(self, config: AgentConfig, api_key: str) -> None:
        self.config = config
        self.api_key = api_key
        self._active: dict[str, subprocess.Popen[str]] = {}
        self._active_lock = threading.Lock()
        self._source_link = self.config.runtime_dir / "source"

    def prepare(self) -> None:
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        requests_dir = self.config.runtime_dir / "requests"
        if requests_dir.is_symlink():
            raise AgentUnavailableError(f"request spool must not be a symlink: {requests_dir}")
        requests_dir.mkdir(parents=True, exist_ok=True)
        for abandoned in requests_dir.iterdir():
            if abandoned.is_dir() and not abandoned.is_symlink():
                shutil.rmtree(abandoned, ignore_errors=True)
            else:
                abandoned.unlink(missing_ok=True)
        self.config.cursor_home.mkdir(parents=True, exist_ok=True)
        for subdir in ("sessions", "chats", "projects"):
            (self.config.cursor_home / "_shared" / subdir).mkdir(
                parents=True,
                exist_ok=True,
            )

        workspace = self.config.workspace_root.expanduser().resolve()
        if not workspace.is_dir():
            raise AgentUnavailableError(f"workspace directory does not exist: {workspace}")

        if self._source_link.is_symlink():
            if self._source_link.resolve() != workspace:
                self._source_link.unlink()
        elif self._source_link.exists():
            raise AgentUnavailableError(
                f"runtime source path exists and is not a symlink: {self._source_link}"
            )
        if not self._source_link.exists():
            self._source_link.symlink_to(workspace, target_is_directory=True)

        self._write_read_only_permissions(self.config.cursor_home)

    def _write_read_only_permissions(
        self,
        home: Path,
        *,
        share_sessions: bool = False,
    ) -> None:
        """Enforce an advisory-only agent even if the mounted source is writable."""
        cursor_dir = home / ".cursor"
        cursor_dir.mkdir(parents=True, exist_ok=True)
        allowed_mcp = self.config.mcp_allowed_servers
        denied_permissions = [
            "Write(**)",
            "Shell(*)",
            "Read(**/.env*)",
            "Read(**/*.key)",
            "Read(**/*.pem)",
            "Read(**/*credential*)",
            "Read(**/*secret*)",
        ]
        if not allowed_mcp:
            denied_permissions.append("Mcp(*:*)")
        policy = {
            "network": {"useHttp1ForAgent": True},
            "permissions": {
                "allow": [
                    "Read(**)",
                    *(f"Mcp({server}:*)" for server in allowed_mcp),
                ],
                "deny": denied_permissions,
            },
        }
        target = cursor_dir / "cli-config.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)

        configured_mcp = json.loads(json.dumps(self.config.mcp_config))
        configured_servers = configured_mcp["mcpServers"]
        configured_mcp["mcpServers"] = {
            server: configured_servers[server] for server in allowed_mcp
        }
        mcp_target = cursor_dir / "mcp.json"
        mcp_temporary = cursor_dir / "mcp.json.tmp"
        mcp_temporary.write_text(
            json.dumps(configured_mcp, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(mcp_temporary, mcp_target)

        if share_sessions:
            shared_root = self.config.cursor_home / "_shared"
            for subdir in ("sessions", "chats", "projects"):
                target = shared_root / subdir
                target.mkdir(parents=True, exist_ok=True)
                link = cursor_dir / subdir
                if not os.path.lexists(link):
                    link.symlink_to(target, target_is_directory=True)

    def binary_path(self) -> str | None:
        binary = self.config.binary
        if os.path.isabs(binary):
            return binary if os.path.isfile(binary) and os.access(binary, os.X_OK) else None
        return shutil.which(binary)

    def ready(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, f"{self.config.api_key_env} is not configured"
        if not self.binary_path():
            return False, f"Cursor Agent binary not found: {self.config.binary}"
        if not self.config.workspace_root.expanduser().is_dir():
            return False, f"workspace directory not found: {self.config.workspace_root}"
        return True, ""

    def _resolve_context_files(self, context_files: list[str]) -> list[Path]:
        workspace = self.config.workspace_root.expanduser().resolve()
        resolved: list[Path] = []
        for supplied in context_files:
            relative = Path(supplied)
            if relative.is_absolute():
                candidate = relative.expanduser().resolve()
            else:
                candidate = (workspace / relative).resolve()
            try:
                candidate.relative_to(workspace)
            except ValueError as exc:
                raise ContextPathError(
                    f"context file must be inside {workspace}: {supplied}"
                ) from exc
            if not candidate.is_file():
                raise ContextPathError(f"context file does not exist: {supplied}")
            relative_to_workspace = candidate.relative_to(workspace)
            resolved.append(self._source_link / relative_to_workspace)
        return resolved

    def _parse_output(self, stdout: str) -> dict[str, Any]:
        stripped = stdout.strip()
        if not stripped:
            raise AgentOutputError("Cursor Agent returned no output")
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise AgentOutputError("Cursor Agent did not return a JSON object")

    def _safe_error(self, stderr: str, stdout: str) -> str:
        message = stderr.strip() or stdout.strip() or "Cursor Agent exited without an error message"
        if self.api_key:
            message = message.replace(self.api_key, "[REDACTED]")
        return message[:2_000]

    def _child_environment(self, home: Path) -> dict[str, str]:
        """Pass networking/runtime variables, but not the service's unrelated secrets."""
        allowed = {
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "NODE_EXTRA_CA_CERTS",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update(
            {
                "CURSOR_API_KEY": self.api_key,
                "HOME": str(home),
                "NO_OPEN_BROWSER": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def _terminate(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                try:
                    process.kill()
                except OSError:
                    pass
            process.wait(timeout=5)

    def shutdown(self) -> None:
        with self._active_lock:
            processes = list(self._active.values())
        for process in processes:
            self._terminate(process)

    def run(
        self,
        prompt: str,
        *,
        model: str,
        session_id: str | None = None,
        context: Any | None = None,
        context_files: list[str] | None = None,
        ephemeral_home: bool = False,
    ) -> AgentResult:
        ready, reason = self.ready()
        if not ready:
            raise AgentUnavailableError(reason)

        prompt_bytes = len(prompt.encode("utf-8"))
        if prompt_bytes > self.config.max_prompt_bytes:
            raise AgentInputTooLargeError(
                f"prompt is {prompt_bytes} bytes; limit is {self.config.max_prompt_bytes}"
            )

        selected_files = self._resolve_context_files(context_files or [])
        request_id = uuid.uuid4().hex
        requests_dir = self.config.runtime_dir / "requests"

        with tempfile.TemporaryDirectory(prefix=f"{request_id}-", dir=requests_dir) as temp:
            request_dir = Path(temp)
            child_home = request_dir / "home"
            child_home.mkdir()
            self._write_read_only_permissions(
                child_home,
                share_sessions=not ephemeral_home,
            )
            prompt_parts = [
                prompt.strip(),
                "",
                "Safety boundary: this is an advisory, read-only task. Do not modify files, "
                "run shell commands, or expose secrets.",
                f"Read-only project source, when needed, is available at {self._source_link}.",
            ]
            if self.config.mcp_allowed_servers:
                servers = ", ".join(self.config.mcp_allowed_servers)
                prompt_parts.append(
                    f"Approved read-only data access is available through MCP: {servers}. "
                    "Query it only when database evidence is materially needed, use bounded "
                    "SELECT queries, and never expose credentials or unrelated rows."
                )

            if context is not None:
                context_text = (
                    context
                    if isinstance(context, str)
                    else json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
                )
                context_bytes = len(context_text.encode("utf-8"))
                if context_bytes > self.config.max_context_bytes:
                    raise AgentInputTooLargeError(
                        f"context is {context_bytes} bytes; limit is "
                        f"{self.config.max_context_bytes}"
                    )
                context_path = request_dir / (
                    "context.txt" if isinstance(context, str) else "context.json"
                )
                context_path.write_text(context_text, encoding="utf-8")
                prompt_parts.append(
                    f"Authoritative request context is stored at {context_path}. Read it fully."
                )

            if selected_files:
                paths = "\n".join(f"- {path}" for path in selected_files)
                prompt_parts.append(f"Additional approved context files:\n{paths}")

            effective_prompt = "\n".join(prompt_parts)
            final_argument = effective_prompt
            if len(effective_prompt.encode("utf-8")) > self.config.prompt_file_threshold_bytes:
                prompt_path = request_dir / "prompt.txt"
                prompt_path.write_text(effective_prompt, encoding="utf-8")
                final_argument = (
                    f"Your complete task is in {prompt_path}. Read that file in full and follow "
                    "it exactly."
                )

            command = [self.config.binary]
            if self.config.agent_endpoint:
                command.extend(["--agent-endpoint", self.config.agent_endpoint])
            command.extend(
                [
                    "--print",
                    "--output-format",
                    "json",
                    "--model",
                    model,
                    "--sandbox",
                    self.config.sandbox,
                    "--workspace",
                    str(self.config.runtime_dir),
                ]
            )
            if self.config.mode != "agent":
                command.extend(["--mode", self.config.mode])
            if self.config.trust_workspace:
                command.append("--trust")
            if self.config.force:
                command.append("--force")
            if session_id:
                command.extend(["--resume", session_id])
            command.append(final_argument)

            started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=self.config.runtime_dir,
                env=self._child_environment(child_home),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                start_new_session=True,
            )
            with self._active_lock:
                self._active[request_id] = process
            try:
                try:
                    stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    self._terminate(process)
                    raise AgentTimeoutError(
                        f"Cursor Agent timed out after {self.config.timeout_seconds} seconds"
                    ) from exc
            finally:
                with self._active_lock:
                    self._active.pop(request_id, None)

            elapsed_ms = max(0, int((time.monotonic() - started) * 1_000))
            if process.returncode != 0:
                raise AgentExecutionError(self._safe_error(stderr, stdout))

            payload = self._parse_output(stdout)
            output = payload.get("result")
            if not isinstance(output, str) or not output.strip():
                raise AgentOutputError("Cursor Agent JSON did not contain a non-empty result")

            usage = payload.get("usage")
            return AgentResult(
                output=output.strip(),
                model=model,
                session_id=(
                    payload.get("session_id")
                    if isinstance(payload.get("session_id"), str)
                    else None
                ),
                request_id=(
                    payload.get("request_id")
                    if isinstance(payload.get("request_id"), str)
                    else None
                ),
                duration_ms=(
                    payload.get("duration_ms")
                    if isinstance(payload.get("duration_ms"), int) and payload["duration_ms"] >= 0
                    else elapsed_ms
                ),
                usage=usage if isinstance(usage, dict) else {},
            )
