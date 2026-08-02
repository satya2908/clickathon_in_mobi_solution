"""ClickHouse access.

A thin wrapper over clickhouse-connect that adds three things the pipeline depends on: a
single place to apply per-query settings, retry on the transient failures a Cloud service
produces during idle wake-up, and a span around every statement so the trace records the exact
SQL each investigation step ran.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import DatabaseError, OperationalError

from .config import ClickHouseConfig

log = logging.getLogger(__name__)

# A Cloud service that has idled down refuses connections until it wakes. These are worth
# retrying; a SQL syntax error is not, and must surface immediately.
_RETRYABLE = (OperationalError, ConnectionError, TimeoutError)

# ClickHouse Cloud suspends an idle service and takes roughly 30 seconds to resume. The
# retry budget has to comfortably exceed that, or the first command after any quiet period
# fails with a connection error that looks like a misconfiguration rather than a cold start.
_CONNECT_ATTEMPTS = 7
_CONNECT_BUDGET_SECONDS = 1 + 2 + 4 + 8 + 15 + 15  # 45s of waiting across the attempts
_QUERY_ATTEMPTS = 3

# ClickHouse Cloud enables async_insert for the default profile. That mode exists to batch many
# small client-side inserts server-side, and it is the wrong one for a bulk load: it cannot
# deduplicate into a dependent materialized view whose inner query aggregates, so pushing a
# million-row block through mv_events_to_5m fails outright with NOT_IMPLEMENTED. Large explicit
# blocks are exactly what sync inserts are for, so the loader asks for them rather than
# disabling dedup in the views and quietly accepting duplicate rollup rows on any retry.
BULK_INSERT_SETTINGS = {"async_insert": 0}


class QueryError(RuntimeError):
    def __init__(self, message: str, sql: str) -> None:
        super().__init__(message)
        self.sql = sql


def as_utc(parameters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stamp naive datetimes as UTC before they are bound.

    The driver reads a naive datetime as *local* time and converts it to UTC on the way out, so
    on a machine at +05:30 a window asked for as midnight arrives at the server as 18:30 the
    previous day. Every bucket column is DateTime('UTC'), so the whole system silently analysed
    a window shifted by the developer's offset: baselines shifted with it, which is why the
    findings still looked sane, but the reported window was wrong, the observed value was mixed
    from two sides of an incident boundary, and the same code produced different answers in
    different timezones.

    Normalising here rather than at each call site because there is no caller for whom local
    time is the right reading. The data has one clock, and it is UTC.
    """
    if not parameters:
        return parameters
    return {key: stamp_utc(value) for key, value in parameters.items()}


def stamp_utc(value: Any) -> Any:
    """Read one naive datetime as UTC, descending into lists. Anything else passes through.

    Lists matter as much as scalars: an ``Array(DateTime)`` parameter is bound element by
    element, so a list left alone here reintroduces the local-time shift above one level down,
    where it is harder to see -- the query runs, returns fewer rows than it should, and reports
    no error.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    if isinstance(value, list):
        return [stamp_utc(item) for item in value]
    if isinstance(value, tuple):
        return tuple(stamp_utc(item) for item in value)
    return value


def rows_as_utc(rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """The same normalisation for values being written rather than matched against.

    Reading was only half the bug. A naive datetime handed to ``insert`` is converted on the
    way out exactly as a query parameter is, so a case investigated at midnight was *stored*
    as 18:30 the previous day on a machine at +05:30. That shift is worse than the read-side
    one it mirrors, because a query returns the wrong rows once while a bad insert is wrong
    for as long as the row exists -- and it made every stored window land on a half-hour
    boundary that no hourly bucket can ever match.
    """
    return [[stamp_utc(value) for value in row] for row in rows]


def render_sql(sql: str, parameters: dict[str, Any] | None) -> str:
    """Substitute bound parameters into a statement so a reader can paste and run it.

    Only ever used for display. Every value here came from the system's own config and its own
    rollup keys rather than from anything a user typed, but rendering is still kept away from
    the execution path on principle: the client binds parameters properly, and the day these two
    diverge should be a day the displayed query looks wrong rather than a day an injected string
    reaches the server.
    """
    out = sql
    for key, value in (parameters or {}).items():
        if isinstance(value, datetime):
            literal = f"'{value:%Y-%m-%d %H:%M:%S}'"
        elif isinstance(value, str):
            literal = "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        elif isinstance(value, bool):
            literal = "1" if value else "0"
        elif value is None:
            literal = "NULL"
        else:
            literal = str(value)
        out = re.sub(r"\{" + re.escape(key) + r":[A-Za-z0-9_()]+\}", literal, out)
    return " ".join(out.split())


class ClickHouse:
    def __init__(self, cfg: ClickHouseConfig, *, tracer: Any | None = None) -> None:
        self.cfg = cfg
        self._tracer = tracer
        self._client: Client | None = None
        # Off by default. A run issues thousands of queries and holding them all would be a
        # slow leak for no benefit; the case file turns it on because provenance is the point
        # there. Keyed by query name so the record is one entry per distinct query *shape*
        # rather than per execution -- a reader wants to know what kinds of question the system
        # asked, not to scroll past four hundred copies of the same one with different keys.
        self.record_sql = False
        self.recorded: dict[str, str] = {}

    def _record(self, name: str, sql: str, parameters: dict[str, Any] | None) -> None:
        if self.record_sql and name not in self.recorded:
            self.recorded[name] = render_sql(sql, parameters)

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self) -> Client:
        last: Exception | None = None
        for attempt in range(_CONNECT_ATTEMPTS):
            try:
                return clickhouse_connect.get_client(
                    host=self.cfg.host,
                    port=self.cfg.port,
                    username=self.cfg.username,
                    password=self.cfg.password,
                    database=self.cfg.database,
                    secure=self.cfg.secure,
                    verify=self.cfg.verify,
                    connect_timeout=self.cfg.connect_timeout,
                    send_receive_timeout=self.cfg.send_receive_timeout,
                    settings=self.cfg.settings,
                )
            except _RETRYABLE as exc:
                last = exc
                wait = min(2**attempt, 15)
                log.warning(
                    "ClickHouse connect failed (attempt %d/%d): %s. A Cloud service that has "
                    "idled down takes about 30 seconds to wake; retrying in %ds",
                    attempt + 1,
                    _CONNECT_ATTEMPTS,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise QueryError(
            f"Could not reach ClickHouse at {self.cfg.host}:{self.cfg.port} after "
            f"{_CONNECT_ATTEMPTS} attempts over roughly {_CONNECT_BUDGET_SECONDS}s. "
            f"Last error: {last}",
            sql="<connect>",
        )

    def _reconnect(self) -> None:
        """Drop a dead connection so the next call re-establishes it.

        A Cloud service that idles down leaves the client holding a socket that looks open and
        fails on first use. Without this the pipeline would abort on the first query after any
        quiet period rather than simply waiting for the service to come back.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - the connection is already broken
                pass
        self._client = None

    def ensure_database(self) -> None:
        """Create the target database, connecting to the server default first.

        The configured database may not exist yet on a fresh Cloud service, and connecting
        straight to a missing database fails before any DDL can run.
        """
        admin = clickhouse_connect.get_client(
            host=self.cfg.host,
            port=self.cfg.port,
            username=self.cfg.username,
            password=self.cfg.password,
            secure=self.cfg.secure,
            verify=self.cfg.verify,
            connect_timeout=self.cfg.connect_timeout,
        )
        try:
            admin.command(f"CREATE DATABASE IF NOT EXISTS {self.cfg.database}")
        finally:
            admin.close()

    @contextmanager
    def _span(self, name: str, sql: str) -> Iterator[None]:
        if self._tracer is None:
            yield
            return
        with self._tracer.span(name, kind="query") as span:
            span.set("db.system", "clickhouse")
            span.set("db.statement", sql[:8000])
            yield

    def _run(self, sql: str, call, *, name: str) -> Any:
        """Execute against the server, reconnecting once if the connection went stale.

        A ``DatabaseError`` -- bad SQL, a missing table, a type mismatch -- is raised
        immediately. Retrying those would turn a clear error into a slow one, and would keep
        re-running statements that will never succeed.
        """
        last: Exception | None = None
        for attempt in range(_QUERY_ATTEMPTS):
            try:
                return call()
            except DatabaseError as exc:
                raise QueryError(str(exc), sql) from exc
            except _RETRYABLE as exc:
                last = exc
                log.warning(
                    "%s failed on a stale or unavailable connection (attempt %d/%d): %s",
                    name,
                    attempt + 1,
                    _QUERY_ATTEMPTS,
                    exc,
                )
                self._reconnect()
                time.sleep(2**attempt)
        raise QueryError(f"{name} failed after {_QUERY_ATTEMPTS} attempts: {last}", sql)

    def query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        name: str = "query",
        settings: dict[str, Any] | None = None,
    ) -> list[tuple]:
        parameters = as_utc(parameters)
        self._record(name, sql, parameters)
        with self._span(name, sql):
            started = time.perf_counter()
            result = self._run(
                sql,
                lambda: self.client.query(sql, parameters=parameters, settings=settings),
                name=name,
            )
            elapsed = (time.perf_counter() - started) * 1000
            log.debug("%s: %d rows in %.0fms", name, len(result.result_rows), elapsed)
            return result.result_rows

    def query_dicts(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        name: str = "query",
    ) -> list[dict[str, Any]]:
        parameters = as_utc(parameters)
        with self._span(name, sql):
            result = self._run(
                sql, lambda: self.client.query(sql, parameters=parameters), name=name
            )
            cols = result.column_names
            return [dict(zip(cols, row, strict=True)) for row in result.result_rows]

    def scalar(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        name: str = "scalar",
        default: Any = None,
    ) -> Any:
        rows = self.query(sql, parameters, name=name)
        if not rows or rows[0][0] is None:
            return default
        return rows[0][0]

    def command(self, sql: str, *, name: str = "command") -> None:
        with self._span(name, sql):
            self._run(sql, lambda: self.client.command(sql), name=name)

    def try_command(self, sql: str, *, name: str = "command") -> bool:
        """Run a statement whose failure is tolerable, reporting success instead of raising.

        For housekeeping that improves the state of the database without being load-bearing:
        OPTIMIZE is the case that matters here. A SummingMergeTree never guarantees collapsed
        parts, and every read in this system sums explicitly rather than assuming one row per
        key, so a merge that could not be scheduled costs some disk and some scan time and
        changes no answer. Aborting a nine-million-row load over it would be the larger error.
        """
        try:
            self.command(sql, name=name)
            return True
        except QueryError as exc:
            log.warning("%s did not run: %s", name, exc)
            return False

    def insert(
        self,
        table: str,
        rows: Sequence[Sequence[Any]],
        column_names: Sequence[str],
        *,
        name: str = "insert",
    ) -> None:
        if not rows:
            return
        statement = f"INSERT INTO {table} ({', '.join(column_names)})"
        stamped = rows_as_utc(rows)
        with self._span(name, statement):
            self._run(
                statement,
                lambda: self.client.insert(
                    table, stamped, column_names=list(column_names), settings=BULK_INSERT_SETTINGS
                ),
                name=name,
            )

    def insert_arrow(self, table: str, arrow_table: Any, *, name: str = "insert_arrow") -> None:
        statement = f"INSERT INTO {table} FORMAT Arrow"
        with self._span(name, statement):
            self._run(
                statement,
                lambda: self.client.insert_arrow(
                    table, arrow_table, settings=BULK_INSERT_SETTINGS
                ),
                name=name,
            )

    def execute_script(self, path: str | Path, *, substitutions: dict[str, Any] | None = None) -> int:
        """Run a semicolon-separated .sql file, returning how many statements executed.

        Statements are split on semicolons that terminate a line so that the parser is not
        confused by semicolons inside string literals in a comment.
        """
        text = Path(path).read_text()
        for key, value in (substitutions or {}).items():
            text = text.replace(f"{{{{{key}}}}}", str(value))

        statements = [s.strip() for s in text.split(";\n") if s.strip()]
        executed = 0
        for statement in statements:
            body = "\n".join(
                line for line in statement.splitlines() if not line.strip().startswith("--")
            ).strip()
            if not body:
                continue
            self.command(body, name="ddl")
            executed += 1
        return executed

    def table_exists(self, table: str) -> bool:
        return bool(
            self.scalar(
                "SELECT count() FROM system.tables WHERE database = {db:String} "
                "AND name = {tbl:String}",
                {"db": self.cfg.database, "tbl": table},
                name="table_exists",
                default=0,
            )
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
