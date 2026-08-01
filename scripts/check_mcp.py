"""Prove the chat surface can actually reach the warehouse.

A model wired to a broken tool does not fail loudly -- it apologises and answers from memory,
which is the single worst outcome available to this project. So the tool gets exercised the way
LibreChat exercises it, over the same SSE transport, before anyone trusts a sentence it produces.

    docker compose --profile chat up -d mcp-clickhouse
    .venv/bin/python scripts/check_mcp.py
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from queue import Empty, Queue

BASE = "http://localhost:8001"
TIMEOUT = 30


def _listen(session: Queue, events: Queue) -> None:
    """Read the SSE stream. Every reply to every request arrives here, not on the POST."""
    with urllib.request.urlopen(f"{BASE}/sse", timeout=TIMEOUT) as stream:
        for raw in stream:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload.startswith("/messages/"):
                session.put(payload)
            else:
                events.put(payload)


def _post(endpoint: str, method: str, params: dict | None = None, rid: int | None = None) -> None:
    body: dict = {"jsonrpc": "2.0", "method": method}
    if rid is not None:
        body["id"] = rid
    if params is not None:
        body["params"] = params
    request = urllib.request.Request(
        f"{BASE}{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(request, timeout=TIMEOUT).read()


def _await(events: Queue, rid: int) -> dict:
    """Replies are interleaved with notifications, so match on id rather than taking the next."""
    while True:
        message = json.loads(events.get(timeout=TIMEOUT))
        if message.get("id") == rid:
            return message


def main() -> int:
    session_q: Queue = Queue()
    events: Queue = Queue()
    threading.Thread(target=_listen, args=(session_q, events), daemon=True).start()

    try:
        endpoint = session_q.get(timeout=10)
    except Empty:
        print(f"No SSE session from {BASE}/sse. Is the mcp-clickhouse container up?")
        return 1

    _post(
        endpoint,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "verdict-check", "version": "1"},
        },
        rid=1,
    )
    server = _await(events, 1)["result"]["serverInfo"]
    _post(endpoint, "notifications/initialized")
    print(f"connected to {server['name']} {server.get('version', '')}".rstrip())

    _post(endpoint, "tools/list", {}, rid=2)
    tools = [t["name"] for t in _await(events, 2)["result"]["tools"]]
    print(f"tools exposed: {', '.join(tools)}\n")

    checks = [
        ("databases visible", "SELECT count() FROM system.databases"),
        ("rollup rows", "SELECT count() FROM rollup_1h"),
        ("cases written", "SELECT count() FROM cases"),
        (
            "fill rate, one real slice",
            "SELECT round(sum(fills) / nullIf(sum(requests), 0), 5) FROM rollup_1h "
            "WHERE combo = 'os_version' AND key_a = 'Android 15' AND key_b = '' "
            "AND bucket >= toDateTime('2026-06-23 00:00:00') "
            "AND bucket < toDateTime('2026-06-26 00:00:00')",
        ),
    ]

    failures = 0
    for rid, (label, sql) in enumerate(checks, start=10):
        _post(endpoint, "tools/call", {"name": "run_select_query", "arguments": {"query": sql}}, rid)
        reply = _await(events, rid)
        text = reply.get("result", {}).get("content", [{}])[0].get("text", "")
        bad = reply.get("result", {}).get("isError") or "error" in text.lower()
        print(f"  {'FAIL' if bad else 'ok  '}  {label:26} {text[:96]}")
        failures += bool(bad)

    print()
    if failures:
        print(f"{failures} check(s) failed. The chat surface would answer from memory.")
        return 1
    print("The chat surface can read the warehouse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
