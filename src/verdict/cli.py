"""Command-line entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Config, ConfigError, load_config
from .db import ClickHouse
from .metrics import MetricRegistry

app = typer.Typer(
    name="verdict",
    help="Automated root-cause analyst for ad-tech metrics on ClickHouse.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
schema_app = typer.Typer(help="Create and inspect the ClickHouse schema.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(schema_app, name="schema")

console = Console()

_SECRET_HINTS = ("password", "api_key", "secret", "token")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("clickhouse_connect").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _redact(key: str, value: object) -> str:
    if any(hint in key.lower() for hint in _SECRET_HINTS) and value:
        return f"<set, {len(str(value))} chars>"
    return str(value)


def _load(config_path: str | None) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error[/]\n{exc}")
        raise typer.Exit(2) from exc


def _registry() -> MetricRegistry:
    return MetricRegistry.load(os.environ.get("VERDICT_METRICS") or "config/metrics.yaml")


@config_app.command("check")
def config_check(
    config: str = typer.Option(None, "--config", "-c", help="Path to verdict.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Validate configuration and the metric registry without touching the network.

    This is also the container healthcheck, so a malformed ConfigMap or a missing secret
    surfaces as an unhealthy container rather than as a run that dies partway through.
    """
    _setup_logging(verbose)
    cfg = _load(config)
    try:
        registry = _registry()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Metric registry error[/]\n{exc}")
        raise typer.Exit(2) from exc

    table = Table(title="Resolved configuration", show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")
    for section, model in cfg.model_dump().items():
        if isinstance(model, dict):
            for key, value in model.items():
                table.add_row(f"{section}.{key}", _redact(key, value))
        else:
            table.add_row(section, _redact(section, model))
    console.print(table)

    console.print(
        f"\n[green]OK[/] {len(registry.metrics)} metrics, "
        f"{len(registry.lattice_dimensions)} lattice dimensions, "
        f"{len(registry.high_cardinality_dimensions)} high-cardinality dimensions"
    )


@config_app.command("matrix")
def config_matrix(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Print the derived metric/dimension validity matrix.

    Worth reading before trusting any slice: it shows which combinations the analyst will
    refuse to compute, and refusing is the point. Fill rate sliced by a post-fill dimension
    returns 1.0 for every value, which is a confident and completely wrong answer.
    """
    _setup_logging(verbose)
    registry = _registry()
    dims = registry.lattice_dimensions

    table = Table(title="Metric x dimension validity", show_header=True, header_style="bold")
    table.add_column("metric")
    for d in dims:
        table.add_column(d.replace("_", "\n"), justify="center")
    for name in registry.metrics:
        row = [name]
        for d in dims:
            row.append("[green]OK[/]" if registry.is_valid_slice(name, d) else "[red]no[/]")
        table.add_row(*row)
    console.print(table)

    console.print("\n[bold]Refusals[/]")
    seen: set[str] = set()
    for name in registry.metrics:
        for d in registry.refused_dimensions(name):
            reason = registry.explain_invalid(name, d)
            if reason not in seen:
                seen.add(reason)
                console.print(f"  [yellow]*[/] {reason}")


@schema_app.command("dump")
def schema_dump(
    config: str = typer.Option(None, "--config", "-c"),
    out: str = typer.Option("sql/generated", "--out", "-o", help="Directory for .sql files"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Write the generated DDL to disk for inspection or manual application."""
    _setup_logging(verbose)
    from .schema import all_statements

    cfg = _load(config)
    registry = _registry()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    statements = all_statements(cfg, registry)
    combined = []
    for i, stmt in enumerate(statements, start=1):
        combined.append(f"-- {i:02d}. {stmt.name}\n{stmt.sql};\n")
    (out_dir / "schema.sql").write_text("\n".join(combined))
    console.print(f"[green]Wrote[/] {len(statements)} statements to {out_dir / 'schema.sql'}")


@schema_app.command("apply")
def schema_apply(
    config: str = typer.Option(None, "--config", "-c"),
    drop: bool = typer.Option(False, "--drop", help="Drop existing objects first"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Create the database, tables, dictionaries, and materialized views."""
    _setup_logging(verbose)
    from .schema import all_statements

    cfg = _load(config)
    registry = _registry()
    ch = ClickHouse(cfg.clickhouse)

    console.print(f"Connecting to [bold]{cfg.clickhouse.host}[/] database [bold]{cfg.clickhouse.database}[/]")
    ch.ensure_database()

    if drop:
        if not typer.confirm(f"Drop every object in {cfg.clickhouse.database}?"):
            raise typer.Abort()
        for obj, kind in [
            ("mv_1h_to_1d", "VIEW"), ("mv_5m_to_1h", "VIEW"), ("mv_events_to_5m", "VIEW"),
            ("dict_apps", "DICTIONARY"), ("dict_advertisers", "DICTIONARY"),
            ("dict_geo_device", "DICTIONARY"),
            ("rollup_5m", "TABLE"), ("rollup_1h", "TABLE"), ("rollup_1d", "TABLE"),
            ("ad_events", "TABLE"), ("dim_apps", "TABLE"), ("dim_advertisers", "TABLE"),
            ("dim_geo_device", "TABLE"), ("cases", "TABLE"), ("case_candidates", "TABLE"),
            ("case_steps", "TABLE"), ("coverage_ledger", "TABLE"), ("feedback", "TABLE"),
            ("runs", "TABLE"),
        ]:
            ch.command(f"DROP {kind} IF EXISTS {obj}", name=f"drop_{obj}")
        console.print("[yellow]Dropped existing objects[/]")

    if cfg.retention.enforce:
        console.print(
            "[yellow]Retention enforcement is ON[/]: raw events older than "
            f"{cfg.retention.raw_events_days} days will be deleted by background merges."
        )

    for stmt in all_statements(cfg, registry):
        ch.command(stmt.sql, name=stmt.name)
        console.print(f"  [green]+[/] {stmt.name}")

    console.print(f"\n[green]Schema applied[/] to {cfg.clickhouse.database}")


@app.command("load")
def load_cmd(
    config: str = typer.Option(None, "--config", "-c"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Overrides run.data_dir"),
    limit: int = typer.Option(None, "--limit", help="Load only the first N fact rows (smoke test)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Load the dataset, then verify it loaded correctly."""
    _setup_logging(verbose)
    from .load import LoadError, load_all

    cfg = _load(config)
    registry = _registry()
    ch = ClickHouse(cfg.clickhouse)
    target = data_dir or cfg.run.data_dir

    try:
        report = load_all(ch, registry, target, limit_rows=limit)
    except LoadError as exc:
        console.print(f"[bold red]Load failed[/]\n{exc}")
        raise typer.Exit(1) from exc

    table = Table(title="Load report", show_header=True, header_style="bold")
    table.add_column("Object")
    table.add_column("Rows", justify="right")
    for name, count in report.dim_rows.items():
        table.add_row(name, f"{count:,}")
    table.add_row("ad_events", f"{report.fact_rows:,}")
    for name, count in report.rollup_rows.items():
        compression = report.fact_rows / count if count else 0
        table.add_row(name, f"{count:,}  ({compression:,.0f}x)")
    console.print(table)

    metrics = Table(title="Global metrics (raw and rollup agree)", header_style="bold")
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")
    for name, value in report.metrics.items():
        metrics.add_row(name, f"{value:,.6g}")
    console.print(metrics)

    console.print(f"\nEvent window: [bold]{report.window[0]}[/] to [bold]{report.window[1]}[/]")
    for warning in report.warnings:
        console.print(f"[yellow]warning[/] {warning}")
    console.print("\n[green]Load verified[/]")


@app.command("version")
def version_cmd() -> None:
    from . import __version__

    console.print(f"verdict {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
