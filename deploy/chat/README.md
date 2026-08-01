# Chat surface

A conversational way to ask the warehouse a question nobody built a panel for. LibreChat runs
the conversation, the ClickHouse MCP server gives the model a tool that runs SQL, and the tool
reads the same rollup tables the detectors read.

## What this is, and what it is not

The model gets a query tool. It does not get a summary of the data, and it does not get any say
in a verdict. Detection, localization and confidence stay entirely in the deterministic
pipeline — switch this whole profile off and every case comes out identical.

That boundary is the point. It answers the follow-up question a fixed dashboard cannot, without
putting a language model anywhere near the decision.

## Running it

```bash
docker compose --profile chat up -d
```

Three containers: `mcp-clickhouse` (the tool), `chat-db` (Mongo, LibreChat's store), and
`librechat` itself on <http://localhost:3080>. First run pulls a few hundred MB.

A model provider key is needed for the chat to answer at all. It reuses `LLM_API_KEY` from
`.env`, the same key the narration layer uses, so there is one credential rather than two.
Without it LibreChat starts, shows its interface, and every message fails.

## Verifying it before trusting it

```bash
.venv/bin/python scripts/check_mcp.py
```

This matters more than it looks. A model wired to a broken tool does not fail loudly — it
apologises and answers from memory, fluently and wrongly. The script drives the tool over the
same SSE transport LibreChat uses and asserts real numbers come back:

```
connected to mcp-clickhouse 1.17.0
tools exposed: list_databases, list_tables, run_select_query
  ok    rollup rows                {"columns":["count()"],"rows":[[1445606]]}
  ok    cases written              {"columns":["count()"],"rows":[[3]]}
  ok    fill rate, one real slice  {"rows":[[0.43331]]}
The chat surface can read the warehouse.
```

Then check LibreChat picked the server up:

```bash
docker logs verdict-librechat 2>&1 | grep MCP
# [MCP][verdict-clickhouse] Tools: list_databases, list_tables, run_select_query
# [MCP] Initialized with 1 configured server and 3 tools.
```

`0 tools` with the server still listed means the address was refused. Add the `host:port` to
`mcpSettings.allowedAddresses` in `config/librechat.yaml` — LibreChat blocks private and local
addresses by default, and the refusal is quiet: the chat works, it just has no tools.

## The bubble

```html
<script src="/chat/bubble.js" data-chat-url="http://localhost:3080"></script>
```

One line, no build step, no framework. It adds a button bottom right and mounts LibreChat in an
iframe on first open. Styling is intentionally minimal — override `.verdict-chat-toggle` and
`.verdict-chat-panel` from the host page's own stylesheet.

`deploy/chat/demo.html` is a host page with nothing else in it, for telling an integration
problem apart from a dashboard problem:

```bash
python3 -m http.server 8090 --directory deploy/chat
open http://localhost:8090/demo.html
```

**Serve the host page over `http://localhost`, not `file://`.** LibreChat keeps a session
cookie, and a page opened from the filesystem makes that cookie third-party, which Chrome drops.
The symptom is a login screen returning on every reload, and it reads as a LibreChat bug rather
than an embedding mistake. Ports are irrelevant — cookies ignore them — so any localhost port
works.

LibreChat sends no `X-Frame-Options` and no `frame-ancestors`, so it frames as shipped. Putting
a reverse proxy in front of it may add those headers and break the bubble; if you do, set
`frame-ancestors 'self' <dashboard-origin>`.

## The one test that proves anything

Ask it: *what was the fill rate for os_version=Android 15 between 23 and 26 June 2026?*

The answer is **0.43331**, and the same figure comes back from `scripts/check_mcp.py` and from
the pipeline. Anything else means the model answered from memory and the tool is not wired.

## Security

This profile is localhost-only by design. The compose file pins `CREDS_KEY`, `JWT_SECRET` and
friends to fixed development values so the stack starts without a setup step, which is exactly
why it must not be exposed beyond the machine it runs on. The MCP server also runs with
`CLICKHOUSE_MCP_AUTH_DISABLED=true` and holds full warehouse credentials.

Before this goes anywhere real: override every secret, enable MCP auth, and give the MCP server
a read-only ClickHouse user rather than `default`.
