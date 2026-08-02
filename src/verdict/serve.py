"""An HTTP front door for the ingest loop, so the console can drive it.

The console is a Node container with no Python in it; the engine is a Python container with no
web server. Something has to bridge them, and the three obvious options are all worse than this
one. Mounting the Docker socket into the web tier so it can `exec` into this one hands a browser-
facing service root on the host. Installing the engine into the web image doubles it and gives
two copies of the analysis code that can drift. Reimplementing ingest in TypeScript guarantees
they drift.

So this runs the CLI, in a subprocess, exactly as an operator would at a terminal. That is the
whole design: there is one implementation of the ingest sequence and this is not it. A shortcut
that skipped the dimension refresh would attribute a whole batch to whatever the dictionaries
last held, which is a bug this project has already shipped once and does not intend to ship
twice.

Deliberately small. No auth, no queue, no streaming: it binds inside a Compose network, runs one
job at a time, and refuses a second while one is running. Exposing it beyond localhost would need
all three, and it is not built for that.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)

#: Bounded so a verbose run cannot grow the response without limit.
MAX_OUTPUT = 64_000

#: Ingest plus an investigation per window. Minutes on a real release.
DEFAULT_TIMEOUT_S = 900


class Runner:
    """Runs one ingest at a time and remembers how the last one went."""

    def __init__(self, root: Path, timeout_s: int = DEFAULT_TIMEOUT_S) -> None:
        self.root = root
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self.busy = False

    def resolve(self, raw: str) -> Path:
        """Resolve a caller-supplied path, or say why it cannot be used.

        ``resolve()`` before the containment test rather than after, so that ``..`` and symlinks
        are followed first. Checking the string and then opening the real file is the standard
        way a containment test gets defeated.
        """
        if not raw or not raw.strip():
            raise ValueError("A path is required")
        candidate = Path(raw.strip())
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if not candidate.exists():
            raise ValueError(f"No such path: {candidate}")

        real = candidate.resolve()
        root = self.root.resolve()
        if real != root and root not in real.parents:
            raise PermissionError(f"Path is outside the permitted root ({root})")

        if real.is_dir() and not (real / "ad_events.parquet").exists():
            raise ValueError(f"{real} does not contain ad_events.parquet")
        if real.is_file() and real.suffix != ".parquet":
            raise ValueError("Expected a directory or a .parquet file")
        return real

    def run(self, target: Path) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("An ingest is already running")
        started = time.monotonic()
        try:
            self.busy = True
            log.info("ingest starting: %s", target)
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, path validated above
                [sys.executable, "-m", "verdict", "ingest", str(target)],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            output = (proc.stdout + proc.stderr)[-MAX_OUTPUT:]
            elapsed = int((time.monotonic() - started) * 1000)
            log.info("ingest finished in %dms with code %s", elapsed, proc.returncode)
            return {
                "ok": proc.returncode == 0,
                "path": str(target),
                "exitCode": proc.returncode,
                "durationMs": elapsed,
                "output": output,
                "error": None if proc.returncode == 0 else f"verdict ingest exited with code {proc.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "path": str(target),
                "exitCode": None,
                "durationMs": int((time.monotonic() - started) * 1000),
                "output": "",
                "error": f"Timed out after {self.timeout_s}s",
            }
        finally:
            self.busy = False
            self._lock.release()


def _handler(runner: Runner) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
            if self.path.rstrip("/") in ("/health", ""):
                self._send(200, {"ok": True, "busy": runner.busy, "root": str(runner.root)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
            if self.path.rstrip("/") != "/ingest":
                self._send(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send(400, {"error": "body must be JSON"})
                return

            try:
                target = runner.resolve(str(body.get("path", "")))
            except PermissionError as exc:
                self._send(403, {"error": str(exc)})
                return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return

            try:
                result = runner.run(target)
            except RuntimeError as exc:
                self._send(409, {"error": str(exc)})
                return

            self._send(200 if result["ok"] else 502, result)

        def log_message(self, fmt: str, *args) -> None:
            log.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def serve(host: str = "0.0.0.0", port: int = 8158, root: str | None = None) -> None:  # noqa: S104
    """Serve the ingest endpoint until interrupted."""
    base = Path(root or os.environ.get("VERDICT_DATA_DIR") or "/data")
    runner = Runner(base)
    server = ThreadingHTTPServer((host, port), _handler(runner))
    log.info("ingest endpoint listening on %s:%d, serving paths under %s", host, port, base)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
