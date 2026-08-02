# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS cursor-agent-source

ENV HOME=/opt/cursor-agent-home \
    PATH=/opt/cursor-agent-home/.local/bin:$PATH \
    SHELL=/bin/bash

RUN apt-get update \
    && apt-get install --no-install-recommends -y bash ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Resolve the official Cursor Agent CLI during the image build rather than committing a binary.
# A corporate root CA, when configured, is a BuildKit secret and never enters an image layer.
RUN --mount=type=secret,id=cursor_extra_ca \
    set -eu; \
    ca_bundle=/etc/ssl/certs/ca-certificates.crt; \
    if [ -s /run/secrets/cursor_extra_ca ] \
        && grep -q "BEGIN CERTIFICATE" /run/secrets/cursor_extra_ca; then \
        cat "$ca_bundle" /run/secrets/cursor_extra_ca >/tmp/cursor-ca-bundle.crt; \
        ca_bundle=/tmp/cursor-ca-bundle.crt; \
    fi; \
    export SSL_CERT_FILE="$ca_bundle" CURL_CA_BUNDLE="$ca_bundle" NODE_EXTRA_CA_CERTS="$ca_bundle"; \
    mkdir -p "${HOME}"; \
    curl --proto '=https' --tlsv1.2 -fsSL https://cursor.com/install \
        -o /tmp/install-cursor-agent.sh \
    && bash /tmp/install-cursor-agent.sh \
    && agent --version \
    && rm -f /tmp/install-cursor-agent.sh /tmp/cursor-ca-bundle.crt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 cursor

COPY --from=cursor-agent-source /opt/cursor-agent-home/ /opt/cursor-agent-home/
RUN ln -s /opt/cursor-agent-home/.local/bin/agent /usr/local/bin/agent \
    && ln -s /opt/cursor-agent-home/.local/bin/cursor-agent /usr/local/bin/cursor-agent \
    && agent --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config.yaml ./config.yaml
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/cursor-service-entrypoint

RUN mkdir -p /workspace /data/runtime /data/jobs /data/cursor-home \
    && chown -R cursor:cursor /app /workspace /data

USER cursor

ENV HOME=/home/cursor \
    CURSOR_SERVICE_CONFIG=/app/config.yaml

EXPOSE 8157

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('CURSOR_SERVICE_PORT', '8157') + '/health', timeout=3)"

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/cursor-service-entrypoint"]
CMD ["python", "-m", "app"]
