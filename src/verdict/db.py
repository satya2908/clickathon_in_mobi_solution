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
        for attempt in range(4):
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
                wait = 2**attempt
                log.warning(
                    "ClickHouse connect failed (attempt %d/4): %s; retrying in %ds",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise QueryError(
            f"Could not reach ClickHouse at {self.cfg.host}:{self.cfg.port} after 4 attempts. "
            f"Last error: {last}",
            sql="<connect>",
        )

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
            try:
                result = self.client.query(sql, parameters=parameters, settings=settings)
            except DatabaseError as exc:
                raise QueryError(str(exc), sql) from exc
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
            try:
                result = self.client.query(sql, parameters=parameters)
            except DatabaseError as exc:
                raise QueryError(str(exc), sql) from exc
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
            try:
                self.client.command(sql)
            except DatabaseError as exc:
                raise QueryError(str(exc), sql) from exc

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
        with self._span(name, f"INSERT INTO {table} ({', '.join(column_names)})"):
            self.client.insert(table, rows, column_names=list(column_names))

    def insert_arrow(self, table: str, arrow_table: Any, *, name: str = "insert_arrow") -> None:
        with self._span(name, f"INSERT INTO {table} FORMAT Arrow"):
            self.client.insert_arrow(table, arrow_table)

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
