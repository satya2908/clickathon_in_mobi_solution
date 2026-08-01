"""ClickHouse access.

A thin wrapper over clickhouse-connect that adds three things the pipeline depends on: a
single place to apply per-query settings, retry on the transient failures a Cloud service
produces during idle wake-up, and a span around every statement so the trace records the exact
SQL each investigation step ran.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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


class ClickHouse:
    def __init__(self, cfg: ClickHouseConfig, *, tracer: Any | None = None) -> None:
        self.cfg = cfg
        self._tracer = tracer
        self._client: Client | None = None

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
        with self._span(name, statement):
            self._run(
                statement,
                lambda: self.client.insert(
                    table, rows, column_names=list(column_names), settings=BULK_INSERT_SETTINGS
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
