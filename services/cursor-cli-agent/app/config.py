"""YAML configuration with small, explicit environment overrides."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigurationError(RuntimeError):
    """Raised when service configuration cannot be loaded safely."""


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8157, ge=1, le=65535)
    max_request_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)


class ModelConfig(BaseModel):
    default: str = "grok"
    remediation: str = "grok"
    validation: str = "grok"
    allow_request_override: bool = False
    allowed: list[str] = Field(default_factory=list)

    @field_validator("default", "remediation", "validation")
    @classmethod
    def model_name_must_be_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model names cannot be empty")
        return value

    @field_validator("allowed")
    @classmethod
    def normalize_allowed_models(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def override_requires_allowlist(self) -> ModelConfig:
        if self.allow_request_override and not self.allowed:
            raise ValueError(
                "models.allowed must not be empty when request model overrides are enabled"
            )
        return self

    def resolve(
        self,
        purpose: Literal["default", "remediation", "validation"],
        requested: str | None = None,
    ) -> str:
        configured = getattr(self, purpose)
        if not requested:
            return configured

        requested = requested.strip()
        if not self.allow_request_override and requested != configured:
            raise ValueError(
                f"request model overrides are disabled; configured {purpose} model is "
                f"{configured!r}"
            )
        if self.allowed and requested not in self.allowed:
            raise ValueError(
                f"model {requested!r} is not allowed; choose one of: {', '.join(self.allowed)}"
            )
        return requested


class AgentConfig(BaseModel):
    binary: str = "agent"
    agent_endpoint: str | None = None
    mcp_config: dict[str, Any] = Field(
        default_factory=lambda: {"mcpServers": {}},
    )
    mcp_allowed_servers: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=300, ge=1, le=86_400)
    max_concurrency: int = Field(default=4, ge=1, le=32)
    max_interactive_requests: int = Field(default=10, ge=1, le=1_000)
    mode: Literal["ask", "plan", "agent"] = "ask"
    trust_workspace: bool = True
    force: bool = False
    sandbox: Literal["enabled", "disabled"] = "enabled"
    workspace_root: Path = Path("/workspace")
    runtime_dir: Path = Path("/data/runtime")
    cursor_home: Path = Path("/data/cursor-home")
    prompt_file_threshold_bytes: int = Field(default=100 * 1024, ge=1024)
    max_prompt_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    max_context_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    api_key_env: str = "CURSOR_API_KEY"

    @field_validator("binary", "api_key_env")
    @classmethod
    def string_must_be_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("agent_endpoint")
    @classmethod
    def normalize_agent_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        secure = parsed.scheme == "https" and bool(parsed.netloc)
        local = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
        if not (secure or local):
            raise ValueError("agent_endpoint must use HTTPS or a local HTTP endpoint")
        return value

    @field_validator("mcp_config")
    @classmethod
    def validate_mcp_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        servers = value.get("mcpServers")
        if not isinstance(servers, dict):
            raise ValueError("mcp_config must contain an mcpServers object")
        for name, server in servers.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("MCP server names must be non-empty strings")
            if not isinstance(server, dict):
                raise ValueError(f"MCP server {name!r} must be an object")
        return value

    @field_validator("mcp_allowed_servers")
    @classmethod
    def normalize_mcp_servers(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @model_validator(mode="after")
    def allowed_mcp_servers_must_be_configured(self) -> AgentConfig:
        configured = set(self.mcp_config["mcpServers"])
        missing = set(self.mcp_allowed_servers) - configured
        if missing:
            raise ValueError(
                "MCP allow-list contains unconfigured servers: " + ", ".join(sorted(missing))
            )
        return self


class RemediationConfig(BaseModel):
    max_recommendations: int = Field(default=5, ge=1, le=10)
    max_case_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    max_pending_jobs: int = Field(default=20, ge=1, le=1_000)


class StorageConfig(BaseModel):
    jobs_dir: Path = Path("/data/jobs")
    retention_days: int = Field(default=30, ge=1, le=3_650)
    max_records: int = Field(default=1_000, ge=10, le=100_000)


class ServiceConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    remediation: RemediationConfig = Field(default_factory=RemediationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def public_dict(self) -> dict[str, Any]:
        """Return non-secret settings that callers may use to render controls."""
        return {
            "models": {
                "default": self.models.default,
                "remediation": self.models.remediation,
                "validation": self.models.validation,
                "allow_request_override": self.models.allow_request_override,
                "allowed": self.models.allowed,
            },
            "agent": {
                "timeout_seconds": self.agent.timeout_seconds,
                "max_concurrency": self.agent.max_concurrency,
                "max_interactive_requests": self.agent.max_interactive_requests,
                "mode": self.agent.mode,
                "sandbox": self.agent.sandbox,
                "agent_endpoint": self.agent.agent_endpoint,
                "mcp_allowed_servers": self.agent.mcp_allowed_servers,
            },
            "remediation": {
                "max_recommendations": self.remediation.max_recommendations,
                "max_pending_jobs": self.remediation.max_pending_jobs,
            },
        }


@dataclass(frozen=True)
class LoadedConfig:
    settings: ServiceConfig
    cursor_api_key: str
    service_api_token: str
    path: Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"expected a boolean, got {value!r}")


def _set(raw: dict[str, Any], section: str, key: str, value: Any) -> None:
    node = raw.setdefault(section, {})
    if not isinstance(node, dict):
        raise ConfigurationError(f"{section!r} must be a YAML mapping")
    node[key] = value


def _apply_environment(raw: dict[str, Any], env: Mapping[str, str]) -> None:
    common_model = env.get("CURSOR_MODEL", "").strip()
    if common_model:
        for key in ("default", "remediation", "validation"):
            _set(raw, "models", key, common_model)

    string_overrides = {
        "CURSOR_DEFAULT_MODEL": ("models", "default"),
        "CURSOR_REMEDIATION_MODEL": ("models", "remediation"),
        "CURSOR_VALIDATION_MODEL": ("models", "validation"),
        "CURSOR_AGENT_BINARY": ("agent", "binary"),
        "CURSOR_AGENT_ENDPOINT": ("agent", "agent_endpoint"),
        "CURSOR_AGENT_MODE": ("agent", "mode"),
        "CURSOR_AGENT_SANDBOX": ("agent", "sandbox"),
        "CURSOR_WORKSPACE_ROOT": ("agent", "workspace_root"),
        "CURSOR_RUNTIME_DIR": ("agent", "runtime_dir"),
        "CURSOR_HOME": ("agent", "cursor_home"),
        "CURSOR_JOBS_DIR": ("storage", "jobs_dir"),
        "CURSOR_SERVICE_HOST": ("server", "host"),
    }
    for name, (section, key) in string_overrides.items():
        value = env.get(name)
        if value is not None and value.strip():
            _set(raw, section, key, value.strip())

    integer_overrides = {
        "CURSOR_SERVICE_PORT": ("server", "port"),
        "CURSOR_MAX_REQUEST_BYTES": ("server", "max_request_bytes"),
        "CURSOR_AGENT_TIMEOUT": ("agent", "timeout_seconds"),
        "CURSOR_MAX_CONCURRENCY": ("agent", "max_concurrency"),
        "CURSOR_MAX_INTERACTIVE_REQUESTS": ("agent", "max_interactive_requests"),
        "CURSOR_MAX_RECOMMENDATIONS": ("remediation", "max_recommendations"),
        "CURSOR_MAX_PENDING_JOBS": ("remediation", "max_pending_jobs"),
        "CURSOR_JOB_RETENTION_DAYS": ("storage", "retention_days"),
        "CURSOR_MAX_JOB_RECORDS": ("storage", "max_records"),
    }
    for name, (section, key) in integer_overrides.items():
        value = env.get(name)
        if value is not None and value.strip():
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer") from exc
            _set(raw, section, key, parsed)

    boolean_overrides = {
        "CURSOR_ALLOW_MODEL_OVERRIDE": ("models", "allow_request_override"),
        "CURSOR_AGENT_TRUST_WORKSPACE": ("agent", "trust_workspace"),
        "CURSOR_AGENT_FORCE": ("agent", "force"),
    }
    for name, (section, key) in boolean_overrides.items():
        value = env.get(name)
        if value is not None and value.strip():
            _set(raw, section, key, _bool(value))

    allowed = env.get("CURSOR_ALLOWED_MODELS")
    if allowed is not None and allowed.strip():
        _set(raw, "models", "allowed", allowed.split(","))

    mcp_config = env.get("CURSOR_MCP_CONFIG")
    if mcp_config is not None and mcp_config.strip():
        if "\n" in mcp_config or "\r" in mcp_config:
            raise ConfigurationError("CURSOR_MCP_CONFIG must be one-line JSON")
        try:
            parsed_mcp = json.loads(mcp_config)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("CURSOR_MCP_CONFIG must be one-line JSON") from exc
        if not isinstance(parsed_mcp, dict):
            raise ConfigurationError("CURSOR_MCP_CONFIG must be a JSON object")
        _set(raw, "agent", "mcp_config", parsed_mcp)

    mcp_allowed = env.get("CURSOR_MCP_ALLOWED_SERVERS")
    if mcp_allowed is not None and mcp_allowed.strip():
        _set(raw, "agent", "mcp_allowed_servers", mcp_allowed.split(","))


def load_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> LoadedConfig:
    source = os.environ if env is None else env
    config_path = Path(
        path or source.get("CURSOR_SERVICE_CONFIG") or DEFAULT_CONFIG_PATH
    ).expanduser()
    if not config_path.exists():
        raise ConfigurationError(f"configuration file not found: {config_path}")

    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"could not read {config_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigurationError("the configuration root must be a YAML mapping")

    raw = dict(parsed)
    _apply_environment(raw, source)
    try:
        settings = ServiceConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(f"invalid configuration in {config_path}: {exc}") from exc

    return LoadedConfig(
        settings=settings,
        cursor_api_key=source.get(settings.agent.api_key_env, "").strip(),
        service_api_token=source.get("CURSOR_SERVICE_API_TOKEN", "").strip(),
        path=config_path.resolve(),
    )
