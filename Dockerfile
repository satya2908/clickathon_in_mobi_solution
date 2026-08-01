# syntax=docker/dockerfile:1.7

# Build stage: resolve dependencies into a self-contained venv so the runtime image carries no
# compiler and no build cache.
FROM python:3.11-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer is keyed on pyproject alone, so editing source does not reinstall the world.
# LICENSE too: pyproject declares `license = { file = "LICENSE" }`, and setuptools reads it
# while generating metadata, so its absence fails the build before a single dependency is
# fetched. It is tiny and never changes, so it costs this layer nothing.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/verdict && touch src/verdict/__init__.py && pip install .

COPY src/ ./src/
RUN pip install --no-deps .


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VERDICT_CONFIG=/etc/verdict/verdict.yaml \
    VERDICT_METRICS=/etc/verdict/metrics.yaml \
    VERDICT_DATA_DIR=/data \
    VERDICT_OUTPUT_DIR=/out

COPY --from=build /opt/venv /opt/venv

# Config lives at a fixed path that a ConfigMap mounts over unchanged. The copy baked into the
# image is the default, not the contract, so an operator overrides it without rebuilding.
COPY config/ /etc/verdict/

RUN useradd --system --uid 10001 --create-home verdict \
    && mkdir -p /data /out \
    && chown -R verdict:verdict /data /out /etc/verdict

USER verdict
WORKDIR /home/verdict

# Fails when the config cannot be parsed or a required variable is unset, so a broken
# ConfigMap surfaces as an unhealthy container rather than as a failed run twenty minutes in.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["verdict", "config", "check"]

ENTRYPOINT ["verdict"]
CMD ["--help"]
