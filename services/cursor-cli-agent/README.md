# Cursor CLI recommendation service

This optional service wraps the official Cursor Agent CLI with a private FastAPI API. Verdict
uses it for two-pass remediation advice: one isolated agent drafts actions, then a second agent
reviews the draft against the deterministic case evidence.

It is integrated into the repository's root Compose file under the `recommendations` profile.
Use the root launcher rather than starting this directory independently:

```bash
./stack.sh up --with-ai
./stack.sh status
```

The core stack does not build or start this image. AI startup requires `CURSOR_API_KEY` in the
root `.env`; `CURSOR_SERVICE_API_TOKEN` is optional and, when set, is passed only between the
Next.js server and this service.

Safety boundaries:

- source paths are mounted read-only;
- the container runs as UID 10001 with all Linux capabilities dropped;
- Cursor runs in ask mode with file writes and shell commands denied;
- `.env`, key, PEM, credential, and secret paths are denied;
- only the read-only `verdict-clickhouse` MCP server is allow-listed;
- Cursor credentials are passed to the child environment, never command arguments or responses.

The image downloads Cursor's official CLI installer at build time. Rebuild the profile to pick
up a newer CLI release:

```bash
./stack.sh rebuild --with-ai
```

If a corporate proxy replaces public TLS certificates, set `CURSOR_CA_CERT` in the root `.env`
to the absolute path of its PEM root certificate. BuildKit injects it as an ephemeral build
secret, and Compose mounts it read-only for Cursor at runtime; the certificate is not vendored.

Local unit tests do not require the Cursor binary:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
ruff check .
```
