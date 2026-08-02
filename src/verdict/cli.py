"""Command-line entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from .config import Config, ConfigError, load_config
from .db import ClickHouse, QueryError
from .metrics import MetricRegistry

if TYPE_CHECKING:
    from .load import LoadReport
    from .query import Window

app = typer.Typer(
    name="verdict",
    help="Autonomous investigator agent for ad-tech metrics on ClickHouse.",
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

    _print_load_report(report, title="Load report")
    console.print("\n[green]Load verified[/]")


def _print_load_report(report: LoadReport, *, title: str) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
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


@app.command("refresh-dimensions")
def refresh_dimensions_cmd(
    config: str = typer.Option(None, "--config", "-c"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Folder holding the dimension CSVs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Swap the dimension tables and rebuild the rollups over the events already loaded.

    For when a release reissues the dimension CSVs with the same IDs and different attribute
    values. The facts are unchanged; only the rollups' copy of the attributes is stale, so this
    re-derives them in place instead of re-uploading the corpus.
    """
    _setup_logging(verbose)
    from .load import LoadError, refresh_dimensions

    cfg = _load(config)
    registry = _registry()
    ch = ClickHouse(cfg.clickhouse)
    target = data_dir or cfg.run.data_dir

    try:
        report = refresh_dimensions(ch, registry, target)
    except LoadError as exc:
        console.print(f"[bold red]Refresh failed[/]\n{exc}")
        raise typer.Exit(1) from exc

    _print_load_report(report, title="Dimensions refreshed")
    console.print(
        "\n[green]Rollups re-derived from the events already loaded[/] — every segment now "
        "means what the new dimension tables say it means, in history as well as in new data."
    )


def _resolve_window(ch: ClickHouse, start: str | None, hours: int, grain: str) -> Window:
    """Work out which window to investigate, defaulting to the most recent complete day.

    Defaulting matters more than it looks. The obvious default -- "now minus 24 hours" -- lands
    on an empty window for a dataset whose events stopped weeks ago, and the run then reports
    nothing wrong with complete confidence. Anchoring on the data's own last bucket means the
    command does something useful with no arguments and never mistakes absent data for calm.
    """
    from datetime import timedelta

    from .query import Window

    if start:
        begin = datetime.fromisoformat(start)
    else:
        rows = ch.query("SELECT max(bucket) FROM rollup_1h", name="latest_bucket")
        latest = rows[0][0] if rows and rows[0][0] is not None else None
        if latest is None:
            console.print("[bold red]No data[/] in rollup_1h. Run 'verdict load' first.")
            raise typer.Exit(1)
        begin = (latest + timedelta(hours=1)) - timedelta(hours=hours)

    return Window(start=begin, end=begin + timedelta(hours=hours), grain=grain)


@app.command("inject")
def inject_cmd(
    out_dir: str = typer.Argument(..., help="Where to write the injected corpus and answer key"),
    config: str = typer.Option(None, "--config", "-c"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Source corpus; overrides run.data_dir"),
    catalogue: str = typer.Option(None, "--catalogue", help="JSON list of specs; omit for the built-in sweep"),
    seed: int = typer.Option(0, "--seed", help="Shifts every segment choice in the built-in sweep"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Plant synthetic incidents in a copy of the corpus and write the answer key beside it.

    The point is to score the system on movements it was never developed against. Accuracy on
    the incidents already in the corpus measures how well the code fits data its author has
    read; the six planted here were chosen for shapes the design is known to handle badly as
    well as ones it should catch, and three of them are expected misses that are recorded as
    known boundaries rather than discovered as surprises. One is a clean control carrying no
    incident at all, where the only correct output is silence.

    Writes a full corpus rather than mutating in place, so the original is never at risk and
    the result loads through the ordinary path: point 'verdict load --data-dir' at the output,
    against a database that is not holding anything you want to keep.
    """
    _setup_logging(verbose)
    import json
    import shutil

    import pyarrow.parquet as pq

    from .inject import DimensionIndex, InjectionError, apply, plan
    from .injectcat import built_in_sweep, specs_from_json

    cfg = _load(config)
    source = Path(data_dir or cfg.run.data_dir)
    target = Path(out_dir)
    fact = source / "ad_events.parquet"
    if not fact.exists():
        console.print(f"[bold red]No corpus[/] at {fact}")
        raise typer.Exit(1)
    if target.resolve() == source.resolve():
        console.print("[bold red]Refusing[/] to write the injected corpus over the original.")
        raise typer.Exit(1)

    target.mkdir(parents=True, exist_ok=True)
    dims = (
        DimensionIndex.from_csv(source / "geo_device.csv", "geo_device_id")
        .merged(DimensionIndex.from_csv(source / "apps.csv", "app_id"))
        .merged(DimensionIndex.from_csv(source / "advertisers.csv", "advertiser_id"))
    )
    for name in ("apps.csv", "advertisers.csv", "geo_device.csv"):
        shutil.copy2(source / name, target / name)

    table = pq.read_table(fact)
    console.print(f"Read [bold]{table.num_rows:,}[/] events from {fact}")

    if catalogue:
        specs = specs_from_json(Path(catalogue).read_text())
    else:
        specs = built_in_sweep(table, dims, seed=seed)

    keys = []
    summary = Table(title="Planted incidents", show_header=True, header_style="bold")
    summary.add_column("Kind")
    summary.add_column("Metric")
    summary.add_column("Segment")
    summary.add_column("Asked", justify="right")
    summary.add_column("Got", justify="right")
    summary.add_column("Rows", justify="right")
    summary.add_column("Expect")

    for spec in specs:
        try:
            incident = plan(spec)
            table, report = apply(table, incident, dimensions=dims)
        except InjectionError as exc:
            console.print(f"[bold red]Injection failed[/] for {spec.kind}/{spec.metric}: {exc}")
            raise typer.Exit(1) from exc

        key = incident.with_outcome(report)
        keys.append(key.to_dict())
        where = ", ".join(f"{k}={v}" for k, v in spec.where.items()) or "(global)"
        expect = "found" if key.expected_detectable else f"miss: {key.blind_spot[:26]}"
        summary.add_row(
            spec.kind,
            spec.metric,
            where[:34],
            f"{spec.magnitude:+.0%}" if spec.magnitude else "-",
            f"{report.realised_magnitude:+.1%}",
            f"{report.rows_changed:,}",
            expect,
        )

    pq.write_table(table, target / "ad_events.parquet")
    (target / "answer_key.json").write_text(json.dumps(keys, indent=2))

    console.print(summary)
    console.print(f"\nWrote corpus and [bold]answer_key.json[/] to {target}")
    console.print(
        "Next: point a throwaway database at it with "
        f"[bold]verdict schema apply[/] then [bold]verdict load --data-dir {target}[/]"
    )


@app.command("ingest")
def ingest_cmd(
    path: str = typer.Argument(..., help="A folder holding ad_events.parquet (and optionally the dimension CSVs), or the parquet itself"),
    config: str = typer.Option(None, "--config", "-c"),
    hours: int = typer.Option(24, "--hours", help="Investigation window size; the batch is tiled into windows this long"),
    grain: str = typer.Option("1h", "--grain", help="Bucket size: 5m, 1h or 1d"),
    metric: list[str] = typer.Option(None, "--metric", "-m", help="Restrict to these metrics; repeatable"),
    no_analyse: bool = typer.Option(False, "--no-analyse", help="Ingest only; do not investigate"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Investigate without writing to ClickHouse"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Force template narration"),
    max_cases: int = typer.Option(25, "--max-cases"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Add a batch of events to the loaded corpus and investigate what arrived.

    The whole loop in one command: append the events without disturbing what is already there,
    refresh the dimensions if the release reissued them, confirm the batch is readable at every
    grain, then investigate each window it covers.
    """
    _setup_logging(verbose)
    from datetime import timedelta

    from .load import LoadError, ingest
    from .query import Window

    cfg = _load(config)
    registry = _registry()
    ch = ClickHouse(cfg.clickhouse)

    try:
        report = ingest(ch, registry, path)
    except (LoadError, QueryError) as exc:
        # QueryError alongside LoadError because the operational failures here are evenly split
        # between the two -- a malformed batch is a LoadError, an unreachable or wrong database
        # is a QueryError -- and only one of them was being caught. The other arrived as a Rich
        # traceback, which is ten kilobytes of box-drawing wrapped around one sentence and is
        # what the console's ingest panel would then show a reader.
        console.print(f"[bold red]Ingest failed[/]\n{_reason(exc)}")
        raise typer.Exit(1) from exc

    size = Table(title="Ingested", header_style="bold")
    size.add_column("")
    size.add_column("", justify="right")
    size.add_row("Events", f"{report.events:,}")
    size.add_row("File", f"{report.file_bytes / 1024 / 1024:.1f} MiB")
    size.add_row("Corpus", f"{report.corpus_before:,} → {report.corpus_after:,}")
    if report.batch_window:
        size.add_row("Window", f"{report.batch_window[0]} to {report.batch_window[1]}")
    if report.dimensions_refreshed:
        size.add_row("Dimensions", f"reissued — rollups re-derived in {report.refresh_s:,.1f}s")
    size.add_row("Read", f"{report.read_s:,.2f}s")
    size.add_row("Insert and roll up", f"{report.insert_s:,.2f}s")
    size.add_row("Verify", f"{report.verify_s:,.2f}s")
    size.add_row(
        "Queryable after",
        f"{report.ingest_s:,.2f}s  ({report.events / report.ingest_s:,.0f} events/s)"
        if report.ingest_s
        else "—",
    )
    console.print(size)
    for warning in report.warnings:
        console.print(f"[yellow]warning[/] {warning}")

    if no_analyse or not report.batch_window:
        return

    # Tile the batch's own span rather than asking for a start time. The point of this command
    # is that the data decides what to look at.
    lo, hi = report.batch_window
    step = timedelta(hours=hours)
    begin = lo.replace(minute=0, second=0, microsecond=0)
    if hours >= 24:
        begin = begin.replace(hour=0)

    windows = []
    while begin <= hi:
        windows.append(Window(start=begin, end=begin + step, grain=grain))
        begin += step

    console.print(
        f"\n[bold]{len(windows)} window(s)[/] to investigate across the batch\n"
    )
    for window in windows:
        _run_window(
            cfg, ch, registry, window,
            metrics=list(metric) if metric else None,
            persist=not no_persist, narrate=not no_llm, max_cases=max_cases,
        )
        console.print()


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host"),  # noqa: S104 - inside a Compose network
    port: int = typer.Option(8158, "--port"),
    root: str = typer.Option(None, "--root", help="Paths outside this are refused. Defaults to VERDICT_DATA_DIR."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Expose `ingest` over HTTP so the console can drive it.

    The console container has no Python and this one has no web server, so the ingest button
    needs somewhere to send its request. This runs the CLI in a subprocess rather than calling
    the pipeline directly, which keeps one implementation of the ingest sequence rather than two.

    Unauthenticated and single-job. It is meant for a Compose network, not an open port.
    """
    _setup_logging(verbose)
    from .serve import serve

    serve(host=host, port=port, root=root)


@app.command("investigate")
def investigate_cmd(
    config: str = typer.Option(None, "--config", "-c"),
    start: str = typer.Option(None, "--start", help="Window start, ISO format. Defaults to the last complete day."),
    hours: int = typer.Option(24, "--hours", help="Window length in hours"),
    grain: str = typer.Option("1h", "--grain", help="Bucket size: 5m, 1h or 1d"),
    metric: list[str] = typer.Option(None, "--metric", "-m", help="Restrict to these metrics; repeatable"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Investigate without writing to ClickHouse"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Force template narration"),
    max_cases: int = typer.Option(25, "--max-cases"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Detect, localize and explain anomalies in one window."""
    _setup_logging(verbose)

    cfg = _load(config)
    registry = _registry()
    ch = ClickHouse(cfg.clickhouse)
    window = _resolve_window(ch, start, hours, grain)
    _run_window(
        cfg, ch, registry, window,
        metrics=list(metric) if metric else None,
        persist=not no_persist, narrate=not no_llm, max_cases=max_cases,
    )


def _run_window(
    cfg: Config,
    ch: ClickHouse,
    registry: MetricRegistry,
    window: Window,
    *,
    metrics: list[str] | None,
    persist: bool,
    narrate: bool,
    max_cases: int,
) -> object:
    """Investigate one window and print everything it concluded."""
    from .pipeline import investigate
    from .trace import Tracer

    no_persist = not persist
    tracer = Tracer(cfg.tracing)

    console.print(f"Investigating [bold]{window.label()}[/] at {window.grain} grain")

    result = investigate(
        cfg,
        ch,
        registry,
        window,
        metrics=metrics,
        tracer=tracer,
        persist=persist,
        narrate=narrate,
        max_cases=max_cases,
    )
    tracer.flush()

    console.print(f"\n{result.summary()}\n")

    # First, because it changes how everything below it should be read.
    if result.temporal_disabled:
        console.print(f"[bold yellow]{result.baseline_audit.detail}[/]\n")

    # Before the all-clear, never after it. An empty case list means one of two opposite
    # things, and the reassuring one must not be printed while the other is true.
    if result.failures:
        console.print(
            f"[bold red]{len(result.failures)} localization(s) failed[/] — the case list below "
            "is incomplete, and a metric missing from it was not necessarily clean:"
        )
        for failure in result.failures:
            console.print(f"  [red]•[/] {failure}")
        console.print()

    if not result.cases:
        if not result.failures:
            console.print("[green]No incident met the reporting bar.[/]")
        _print_coverage(result)
        return result

    table = Table(title="Cases", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Verdict")
    table.add_column("Segment")
    table.add_column("Observed", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Effect", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Impact", justify="right")

    for case in result.cases:
        row = case.case_row()
        observed, expected, effect = row[8], row[9], row[10]
        impact = case.impact
        money = "" if impact.revenue is None else f"{impact.revenue:,.0f}{'' if impact.direct else '*'}"
        table.add_row(
            case.finding.metric,
            case.verdict_kind,
            case.segment.label(),
            f"{observed:,.5g}",
            f"{expected:,.5g}",
            f"{effect:+.1%}",
            f"{case.confidence_value:.2f}",
            money,
        )
    console.print(table)
    console.print("[dim]* revenue reached through a chain of estimates; see impact_json for the basis.[/]")

    _print_coverage(result)

    if result.persisted:
        console.print(f"\n[green]Persisted[/] as run {result.run_id}")
    elif not no_persist:
        console.print("\n[yellow]One or more writes failed; see logs.[/]")
    return result


def _print_coverage(result: object) -> None:
    """Report what could not be tested, always, including when nothing was found.

    Printed unconditionally because a clean run and a run that could not look are the two
    outcomes most easily confused, and only one of them is good news.
    """
    gaps = getattr(result, "gaps", [])
    if not gaps:
        return
    by_reason: dict[str, int] = {}
    for gap in gaps:
        by_reason[gap.reason] = by_reason.get(gap.reason, 0) + 1
    detail = ", ".join(f"{count:,} {reason}" for reason, count in sorted(by_reason.items()))
    console.print(f"[yellow]Coverage:[/] {len(gaps):,} cell(s) could not be tested ({detail})")


@app.command("version")
def version_cmd() -> None:
    from . import __version__

    console.print(f"verdict {__version__}")


def _reason(exc: Exception) -> str:
    """The sentence in an exception, without the stack that carried it.

    ClickHouse errors arrive with the driver's URL, version banner and SQL appended, which is
    useful in a log and noise in a console panel. The first line is the part that says what went
    wrong.
    """
    first = str(exc).strip().splitlines()
    return first[0].strip() if first else exc.__class__.__name__


def main() -> None:
    app()


if __name__ == "__main__":
    main()
