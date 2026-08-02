"""Configuration loading.

Config is plain YAML with ``${VAR}`` / ``${VAR:-default}`` placeholders resolved from the
environment at load time. Nothing secret is ever written into the YAML, so the same file is
valid as a Kubernetes ConfigMap, a Docker bind mount, or a local file, with credentials
arriving separately as env vars (a Secret, in Kubernetes).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_PLACEHOLDER = re.compile(
    r"""\$\{
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        (?:
            (?P<op>:-|-)
            (?P<default>[^}]*)
        )?
    \}""",
    re.VERBOSE,
)

# Deliberately excludes "0" and "1". In configuration those are overwhelmingly numbers --
# max_threads, replica counts, retry budgets -- and treating them as booleans turns
# ${MAX_THREADS:-0} into False, which ClickHouse then tries to parse as a setting value and
# rejects with an error that points nowhere near the cause.
_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


def _substitute(text: str, source: dict[str, str]) -> str:
    """Resolve every ``${VAR}`` placeholder in ``text``.

    ``${VAR}`` is required and raises when unset. ``${VAR:-default}`` falls back when the
    variable is unset *or* empty; ``${VAR-default}`` falls back only when it is unset, which
    matches POSIX shell semantics so that operators can reason about it without surprises.
    """
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        op = match.group("op")
        default = match.group("default")
        if name in source:
            value = source[name]
            if value == "" and op == ":-":
                return default
            return value
        if op is not None:
            return default
        missing.append(name)
        return ""

    result = _PLACEHOLDER.sub(replace, text)
    if missing:
        raise ConfigError(
            "Required environment variable(s) not set: "
            + ", ".join(sorted(set(missing)))
            + ". Set them in the environment (or a Kubernetes Secret) before starting."
        )
    return result


def _coerce(value: str) -> Any:
    """Re-type a substituted scalar.

    Substitution happens on raw text before YAML parsing would normally assign types, so
    ``${PORT:-8443}`` would otherwise arrive as the string ``"8443"``. Pydantic would coerce
    most of these anyway, but doing it here keeps raw dict access honest too.
    """
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def expand(node: Any, source: dict[str, str] | None = None) -> Any:
    """Recursively resolve placeholders through a parsed YAML structure."""
    src = dict(os.environ) if source is None else source
    if isinstance(node, dict):
        return {k: expand(v, src) for k, v in node.items()}
    if isinstance(node, list):
        return [expand(v, src) for v in node]
    if isinstance(node, str):
        if not _PLACEHOLDER.search(node):
            return node
        expanded = _substitute(node, src)
        # Only re-type when the placeholder was the whole value. "db_${ENV}" stays a string.
        if _PLACEHOLDER.fullmatch(node.strip()):
            return _coerce(expanded)
        return expanded
    return node


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``, recursing into dicts and replacing everything else."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ClickHouseConfig(BaseModel):
    host: str
    port: int = 8443
    username: str = "default"
    password: str = ""
    database: str = "verdict"
    secure: bool = True
    verify: bool = True
    connect_timeout: int = 30
    send_receive_timeout: int = 300
    # How the lattice is read. "batch" fetches it in one query and answers the scan from
    # memory; "per_combo" issues a query per combination. Identical results either way -- the
    # difference is how many round trips the run spends, which on a remote service is most of
    # its wall time. See RollupReader for the full argument.
    read_mode: Literal["batch", "per_combo"] = "batch"
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_host(self) -> ClickHouseConfig:
        if not self.host or self.host.startswith("${"):
            raise ConfigError("clickhouse.host is unset; export CLICKHOUSE_HOST")
        if self.secure and self.port == 8123:
            raise ConfigError(
                "clickhouse.secure is true but port is 8123 (the plaintext port). "
                "Use 8443 for TLS, or set CLICKHOUSE_SECURE=false."
            )
        return self


class LLMConfig(BaseModel):
    """Any OpenAI-compatible endpoint. Narration is optional by construction: when disabled,
    the pipeline emits template prose and every number in a case file is unchanged."""

    enabled: bool = True
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_seconds: int = 60
    max_retries: int = 2
    # Sent only when set, because providers disagree about it: Gemini accepts "low" and rejects
    # "none" with a 400, and an endpoint that has never heard of the parameter should not be made
    # to reject every request. Worth having because reasoning models bill invisible thinking
    # against max_tokens, and a budget spent on thinking returns prose that stops mid-sentence.
    reasoning_effort: str = ""

    @model_validator(mode="after")
    def _check_key(self) -> LLMConfig:
        if self.enabled and not self.api_key:
            raise ConfigError(
                "llm.enabled is true but no API key is set. Export LLM_API_KEY, or set "
                "LLM_ENABLED=false to run fully deterministically with template narration."
            )
        return self


class TracingConfig(BaseModel):
    """OpenTelemetry export to ClickStack/HyperDX. Every investigation step is a span, so the
    trace doubles as the audit log and as the reviewable record of what was tried."""

    enabled: bool = True
    endpoint: str = "http://localhost:4318"
    service_name: str = "verdict"
    api_key: str = ""
    console_fallback: bool = False


class DetectionConfig(BaseModel):
    baseline_weeks: int = 4
    min_baseline_samples: int = 2
    trim_extremes: bool = True
    dispersion_floor: float = 1.0
    dispersion_ceiling: float = 50.0
    p_value_threshold: float = 0.01
    min_relative_effect: float = 0.05
    detect_rises: bool = True
    grains: list[str] = Field(default_factory=lambda: ["5m", "1h", "1d"])
    target_power: float = 0.80
    structural_min_levels: int = 3
    structural_z_threshold: float = 5.0
    structural_min_window_days: int = 1

    # Whether to check that the temporal baseline still describes the population before
    # trusting anything it says. See verdict/baseline.py.
    baseline_audit_enabled: bool = True
    baseline_audit_windows: int = 2
    # Share of tested cells a calibrated baseline may flag on a recent window. A correct
    # baseline flags roughly the false-discovery rate plus whatever genuinely happened, so
    # single digits. Ten percent is generous; the failure this catches sits above forty.
    baseline_audit_max_flagged: float = 0.10


class LocalizationConfig(BaseModel):
    max_candidates: int = 40
    max_depth: int = 2
    # Relative movement below which a parent metric is treated as not having moved, used only
    # when the metric's own significance test cannot run on the parent series. That happens for
    # continuous ratios early in a corpus, where fewer than three prior weeks exist to measure a
    # spread against. Set well under any business-meaningful move and well over numerical noise;
    # its only job is to separate "flat by construction" from "moved a little".
    parent_moved_floor: float = 0.005
    sufficiency_threshold: float = 0.60
    minimality_threshold: float = 0.30
    maximality_threshold: float = 0.50
    exoneration_residual_tolerance: float = 0.005
    holdout_enabled: bool = True


class ConfidenceConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "significance": 0.25,
            "sufficiency": 0.30,
            "minimality": 0.15,
            "stability": 0.15,
            "separation": 0.15,
        }
    )
    publish_threshold: float = 0.50

    @model_validator(mode="after")
    def _check_weights(self) -> ConfidenceConfig:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"confidence.weights must sum to 1.0, got {total:.6f}")
        return self


class RetentionConfig(BaseModel):
    """Tiered retention. The day counts describe production intent; ``enforce`` decides whether
    ClickHouse actually applies them as TTL clauses.

    ``enforce`` defaults to false because a historical dataset loaded for analysis is, by
    definition, older than a 7-day raw TTL. Turning TTL on by default would hand ClickHouse a
    correct instruction to delete the entire corpus on the first background merge, some
    minutes after a load that appeared to succeed.
    """

    enforce: bool = False
    raw_events_days: int = 7
    rollup_5m_days: int = 30
    rollup_1h_days: int = 180
    rollup_1d_days: int = 730
    cases_days: int = 730


class RunConfig(BaseModel):
    """Where the pipeline reads from and writes to."""

    data_dir: str = "./data"
    output_dir: str = "./out"
    replay_speed: float = 0.0


class Config(BaseModel):
    clickhouse: ClickHouseConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    localization: LocalizationConfig = Field(default_factory=LocalizationConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def redacted(self) -> dict[str, Any]:
        """The configuration with every credential removed.

        A run row records the settings a verdict was produced under, which is what makes an old
        case interpretable once thresholds have moved on. That row lives in a table any reader
        can query, so the database password and the model API key must not travel with it.

        Redaction is by field name rather than by an allow-list of safe fields, so a credential
        added later is covered by default. The name is preserved and only the value is replaced,
        because knowing that a key was configured is itself useful when explaining why a case has
        no narration.
        """
        secret_names = {"password", "api_key", "secret", "token"}
        data = self.model_dump(mode="json")
        for section in data.values():
            if not isinstance(section, dict):
                continue
            for key in list(section):
                if key in secret_names and section[key]:
                    section[key] = "***"
        return data

    def redacted_json(self) -> str:
        return json.dumps(self.redacted(), sort_keys=True, separators=(",", ":"))


DEFAULT_CONFIG_PATH = Path("config/verdict.yaml")
DEFAULT_ENV_FILE = Path(".env")


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a plain dict. Missing file is not an error.

    Written out rather than pulling in python-dotenv: the format is a handful of lines and the
    precedence rules matter more than the parsing, so they belong somewhere visible.

    Deliberately does not strip trailing comments from unquoted values. Implementations disagree
    on whether ``KEY=a#b`` holds ``a#b`` or ``a``, and a credential silently truncated at a ``#``
    is a miserable thing to debug. Wrap a value in quotes if it needs to end in whitespace.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_config(
    path: str | Path | None = None,
    overlay: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Load and validate configuration.

    Resolution order: explicit ``path`` argument, then ``$VERDICT_CONFIG``, then
    ``config/verdict.yaml``. An overlay file (``$VERDICT_CONFIG_OVERLAY``) is deep-merged on
    top when present, which is how a per-environment ConfigMap layers onto the shipped base.

    Settings come from ``.env`` first and the real environment second, the environment winning
    every collision. Reading the file at all is what stops ``.env`` being a Docker-only artefact:
    Compose loads it automatically, so editing a value there and running the CLI directly used to
    change nothing, with no error to explain the silence. Letting the environment win keeps that
    honest in the other direction -- a Kubernetes secret or an explicit ``LLM_ENABLED=false`` on
    the command line is never shadowed by a file someone left in the working directory.

    Passing ``env`` explicitly skips the file entirely, so tests stay hermetic.
    """
    if env is None:
        src = {**read_dotenv(Path(os.environ.get("VERDICT_ENV_FILE", DEFAULT_ENV_FILE))), **os.environ}
    else:
        src = env
    base_path = Path(path or src.get("VERDICT_CONFIG") or DEFAULT_CONFIG_PATH)
    if not base_path.exists():
        raise ConfigError(
            f"Config file not found at {base_path}. Copy config/verdict.yaml or set "
            "VERDICT_CONFIG to its location."
        )

    raw = yaml.safe_load(base_path.read_text()) or {}

    overlay_path = overlay or src.get("VERDICT_CONFIG_OVERLAY")
    if overlay_path and Path(overlay_path).exists():
        raw = deep_merge(raw, yaml.safe_load(Path(overlay_path).read_text()) or {})

    return Config.model_validate(expand(raw, src))
