"""The deployment manifests have to invoke a CLI that exists.

A manifest is code that nothing type-checks and no test exercises, so a flag renamed in cli.py
leaves the scheduled path broken and silent until a CronJob fails in a cluster at 3am. That is
exactly what happened here: `investigate --since 2h` was committed against a CLI that only ever
had --start and --hours, and it exited 2 every time it fired.

These tests parse the args out of the YAML and hand them to Typer's own parser, so the contract
being checked is the real one rather than a list of flag names kept in step by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from verdict.cli import app

MANIFESTS = sorted((Path(__file__).parent.parent / "deploy" / "k8s").glob("*.yaml"))
runner = CliRunner()


def _containers(doc: dict) -> list[dict]:
    """Every container in a Job, CronJob or Deployment, wherever the spec nests it."""
    if not isinstance(doc, dict):
        return []
    found: list[dict] = []
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("containers", "initContainers") and isinstance(value, list):
                    found.extend(c for c in value if isinstance(c, dict))
                else:
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _invocations() -> list[tuple[str, str, list[str]]]:
    out = []
    for path in MANIFESTS:
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            for container in _containers(doc or {}):
                args = container.get("args")
                if args:
                    out.append((path.name, container.get("name", "?"), [str(a) for a in args]))
    return out


INVOCATIONS = _invocations()


def test_the_manifests_were_found_at_all():
    """Without this the parametrised tests below pass by having nothing to check."""
    assert INVOCATIONS, f"no container args found under {MANIFESTS}"


@pytest.mark.parametrize(
    ("manifest", "container", "args"),
    INVOCATIONS,
    ids=[f"{m}:{c}" for m, c, _ in INVOCATIONS],
)
def test_every_manifest_invocation_parses(manifest, container, args):
    """--help exits 0 once the command and its options are recognised, and 2 when they are not.

    Parsing only: running these for real would need a database. The failure being caught is a
    manifest that cannot start, which surfaces at parse time.
    """
    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, (
        f"{manifest} container {container!r} runs `verdict {' '.join(args)}`, "
        f"which the CLI rejects:\n{result.output}"
    )
