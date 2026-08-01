# The console. Reads ClickHouse directly from server components; there is no API tier between
# them, because a second service would exist only to forward the same query with the same
# credentials to the same database.
#
# Three stages so the runtime image carries neither the toolchain nor the dev dependencies.

FROM node:24-alpine AS deps
WORKDIR /app

# Optional corporate root CA, for networks that intercept TLS. Drop a .crt into web/certs/ and
# the install trusts it; on an ordinary network the directory is absent and this is a no-op.
# The directory is gitignored, because a build must not depend on one company's certificate.
#
# `package.json` is listed first so the COPY always resolves: a wildcard matching nothing is a
# build failure, and `certs*/` matches nothing when the directory does not exist.
COPY package.json certs*/ /tmp/ca/
# NODE_EXTRA_CA_CERTS extends Node's built-in root store, so nothing needs installing --
# which matters, because `apk add ca-certificates` would itself need the CA it is fetching.
# When no cert was supplied the file is seeded from the system bundle: a valid but redundant
# PEM, which keeps Node quiet where an empty file would make it warn on every invocation.
RUN if ls /tmp/ca/*.crt >/dev/null 2>&1; then \
      cat /tmp/ca/*.crt > /usr/local/share/extra-ca.crt && \
      echo "trusting $(grep -c 'BEGIN CERTIFICATE' /usr/local/share/extra-ca.crt) extra CA certificate(s)"; \
    else \
      cp /etc/ssl/certs/ca-certificates.crt /usr/local/share/extra-ca.crt 2>/dev/null || \
      cp /etc/ssl/cert.pem /usr/local/share/extra-ca.crt; \
    fi
ENV NODE_EXTRA_CA_CERTS=/usr/local/share/extra-ca.crt

COPY package.json package-lock.json* ./

# `npm ci` rather than `npm install`, so an image build cannot silently resolve a different
# dependency tree than the one that was tested. The committed lockfile resolves to public npm
# regardless of what registry the developer installed through -- see scripts/public-lockfile.mjs,
# which runs on postinstall and is the reason this line needs no registry flag.
#
# `--ignore-scripts` because the only lifecycle hook here is the lockfile normaliser, which
# lives in scripts/ and is not copied into this stage -- and would have nothing to do anyway,
# since the lockfile it rewrites is discarded with the layer.
RUN --mount=type=cache,target=/root/.npm \
    if [ -f package-lock.json ]; then npm ci --no-audit --no-fund --ignore-scripts; \
    else npm install --no-audit --no-fund --ignore-scripts; fi

FROM node:24-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# node_modules is already installed; re-running the postinstall hook here would only rewrite a
# lockfile that is about to be discarded.
ENV npm_config_ignore_scripts=true
# The page is `force-dynamic`, so nothing here reaches the database: the build produces a
# server bundle, not rendered output. That is deliberate -- a root-cause console must not
# serve a snapshot of the world as it looked at build time.
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:24-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0

# `output: standalone` emits a server with only the modules it actually reached, so the
# runtime layer does not need node_modules at all.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static

RUN addgroup -S app && adduser -S app -G app && chown -R app:app /app
USER app

EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=5 \
  CMD wget -qO- http://127.0.0.1:3000/ >/dev/null 2>&1 || exit 1

CMD ["node", "server.js"]
