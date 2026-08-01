# The console. Reads ClickHouse directly from server components; there is no API tier between
# them, because a second service would exist only to forward the same query with the same
# credentials to the same database.
#
# Three stages so the runtime image carries neither the toolchain nor the dev dependencies.

FROM node:24-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
# `npm ci` when the lockfile is present, so an image build cannot silently resolve a different
# dependency tree than the one that was tested.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

FROM node:24-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
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
COPY --from=build /app/public ./public

RUN addgroup -S app && adduser -S app -G app && chown -R app:app /app
USER app

EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=5 \
  CMD wget -qO- http://127.0.0.1:3000/ >/dev/null 2>&1 || exit 1

CMD ["node", "server.js"]
