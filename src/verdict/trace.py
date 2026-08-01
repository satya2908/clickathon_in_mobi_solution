"""Investigation tracing.

Every step the analyst takes opens a span carrying three fields: what was run, why it was run,
and what came back. That triple is not decoration -- it is the entire explanation surface. The
same records feed three consumers, which is why they are captured once:

  * ClickStack / HyperDX, where a reviewer can walk the investigation as a trace waterfall
  * the ``case_steps`` table, so a case file can be rebuilt long after the trace expires
  * the drill-down panel in the HTML report

Because all three read one record, the story a reviewer is told and the story the audit log
holds cannot drift apart. If a step was skipped, the gap is visible in all three.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import TracingConfig

log = logging.getLogger(__name__)


@dataclass
class Step:
    """One recorded investigation step."""

    step_id: str
    parent_id: str
    ordinal: int
    name: str
    kind: str
    what: str = ""
    why: str = ""
    result: str = ""
    sql: str = ""
    duration_ms: int = 0
    span_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("attributes")
        return d


class Span:
    """Handle passed to the body of a traced step."""

    def __init__(self, step: Step, otel_span: Any | None) -> None:
        self._step = step
        self._otel = otel_span

    @property
    def step(self) -> Step:
        return self._step

    def set(self, key: str, value: Any) -> Span:
        self._step.attributes[key] = value
        if self._otel is not None:
            self._otel.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
        if key == "db.statement":
            self._step.sql = str(value)
        return self

    def what(self, text: str) -> Span:
        """The action taken, in plain language."""
        self._step.what = text
        return self.set("verdict.what", text)

    def why(self, text: str) -> Span:
        """The reason this step was worth running, stated before its result is known."""
        self._step.why = text
        return self.set("verdict.why", text)

    def result(self, text: str) -> Span:
        """What came back, including when the answer was 'nothing'."""
        self._step.result = text
        return self.set("verdict.result", text)


class Tracer:
    def __init__(self, cfg: TracingConfig, *, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.run_id = run_id or uuid.uuid4().hex
        self.steps: list[Step] = []
        self._stack: list[str] = []
        self._ordinal = 0
        self._otel_tracer: Any | None = None
        self._provider: Any | None = None
        if cfg.enabled:
            self._init_otel()

    def _init_otel(self) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError:
            log.warning("OpenTelemetry not installed; investigation steps recorded in-memory only")
            return

        resource = Resource.create(
            {"service.name": self.cfg.service_name, "verdict.run_id": self.run_id}
        )
        provider = TracerProvider(resource=resource)

        headers = {"authorization": self.cfg.api_key} if self.cfg.api_key else None
        try:
            exporter = OTLPSpanExporter(
                endpoint=f"{self.cfg.endpoint.rstrip('/')}/v1/traces", headers=headers
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception as exc:  # noqa: BLE001 - tracing must never break the pipeline
            log.warning("OTLP exporter unavailable (%s); continuing without export", exc)

        if self.cfg.console_fallback:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        self._provider = provider
        self._otel_tracer = provider.get_tracer("verdict")

    @contextmanager
    def span(self, name: str, *, kind: str = "step") -> Iterator[Span]:
        self._ordinal += 1
        step = Step(
            step_id=uuid.uuid4().hex[:16],
            parent_id=self._stack[-1] if self._stack else "",
            ordinal=self._ordinal,
            name=name,
            kind=kind,
        )
        self.steps.append(step)
        self._stack.append(step.step_id)
        started = time.perf_counter()

        if self._otel_tracer is None:
            try:
                yield Span(step, None)
            finally:
                step.duration_ms = int((time.perf_counter() - started) * 1000)
                self._stack.pop()
            return

        with self._otel_tracer.start_as_current_span(name) as otel_span:
            step.span_id = format(otel_span.get_span_context().span_id, "016x")
            try:
                yield Span(step, otel_span)
            except Exception as exc:
                otel_span.record_exception(exc)
                step.result = f"failed: {exc}"
                raise
            finally:
                step.duration_ms = int((time.perf_counter() - started) * 1000)
                self._stack.pop()

    @property
    def trace_id(self) -> str:
        """The trace id of the first recorded span, used to deep-link a case into HyperDX."""
        for step in self.steps:
            if step.span_id:
                return step.span_id
        return ""

    def steps_for_case(self, case_id: str) -> list[dict[str, Any]]:
        return [{"case_id": case_id, **s.as_row()} for s in self.steps]

    def reset(self) -> None:
        """Clear recorded steps between cases while keeping the exporter alive."""
        self.steps = []
        self._stack = []
        self._ordinal = 0

    def flush(self) -> None:
        if self._provider is not None:
            try:
                self._provider.force_flush(timeout_millis=5000)
            except Exception as exc:  # noqa: BLE001
                log.warning("Trace flush failed: %s", exc)

    def shutdown(self) -> None:
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                log.warning("Trace shutdown failed: %s", exc)


class NullTracer(Tracer):
    """Tracer that records in memory but never exports. Used in tests."""

    def __init__(self) -> None:
        super().__init__(TracingConfig(enabled=False))
