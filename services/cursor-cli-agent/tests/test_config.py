from pathlib import Path

import pytest

from app.config import ConfigurationError, load_config


def test_models_can_be_selected_from_yaml_and_environment(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
models:
  default: yaml-default
  remediation: yaml-generation
  validation: yaml-validation
""",
        encoding="utf-8",
    )

    loaded = load_config(
        config,
        env={
            "CURSOR_API_KEY": "secret",
            "CURSOR_AGENT_ENDPOINT": "https://api2.cursor.sh/",
            "CURSOR_MCP_CONFIG": (
                '{"mcpServers":{"verdict-clickhouse":'
                '{"url":"http://mcp-clickhouse:8000/sse"}}}'
            ),
            "CURSOR_MCP_ALLOWED_SERVERS": "verdict-clickhouse",
            "CURSOR_VALIDATION_MODEL": "environment-validation",
        },
    )

    assert loaded.settings.models.default == "yaml-default"
    assert loaded.settings.models.remediation == "yaml-generation"
    assert loaded.settings.models.validation == "environment-validation"
    assert loaded.settings.agent.agent_endpoint == "https://api2.cursor.sh"
    assert loaded.settings.agent.mcp_allowed_servers == ["verdict-clickhouse"]
    assert (
        loaded.settings.agent.mcp_config["mcpServers"]["verdict-clickhouse"]["url"]
        == "http://mcp-clickhouse:8000/sse"
    )
    assert loaded.cursor_api_key == "secret"


def test_common_model_environment_value_sets_all_purposes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    loaded = load_config(config, env={"CURSOR_MODEL": "one-model"})

    assert loaded.settings.models.default == "one-model"
    assert loaded.settings.models.remediation == "one-model"
    assert loaded.settings.models.validation == "one-model"


def test_request_override_requires_explicit_configuration(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    models = load_config(config, env={}).settings.models

    assert models.resolve("default") == "grok"
    with pytest.raises(ValueError, match="overrides are disabled"):
        models.resolve("default", "different")


def test_invalid_boolean_override_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="expected a boolean"):
        load_config(config, env={"CURSOR_AGENT_FORCE": "sometimes"})


def test_mcp_config_environment_value_must_be_one_line(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="one-line JSON"):
        load_config(
            config,
            env={"CURSOR_MCP_CONFIG": '{\n"mcpServers": {}\n}'},
        )


def test_model_override_fails_closed_without_allowlist(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="models.allowed"):
        load_config(config, env={"CURSOR_ALLOW_MODEL_OVERRIDE": "true"})


def test_model_override_honors_allowlist(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    models = load_config(
        config,
        env={
            "CURSOR_ALLOW_MODEL_OVERRIDE": "true",
            "CURSOR_ALLOWED_MODELS": "grok,gpt-5",
        },
    ).settings.models

    assert models.resolve("default", "gpt-5") == "gpt-5"
    with pytest.raises(ValueError, match="not allowed"):
        models.resolve("default", "unapproved")
