/** Where the trace UI lives. Two deployments are supported and they need different URLs: the
 *  hosted HyperDX that ships with ClickHouse Cloud, and a self-hosted ClickStack container on
 *  8080. `NEXT_PUBLIC_` because the link is clicked in the browser, not during render.
 *
 *  Defaults to the hosted one, which is what runs today -- the `clickstack` service exists in
 *  the compose file but sits behind the `selfhosted` profile and is not started. */
const BASE = (process.env.NEXT_PUBLIC_HYPERDX_URL || 'https://hyperdx.clickhouse.cloud').replace(/\/+$/, '');

/** The trace source id, if the deployment has more than one and the default is wrong.
 *  Optional: HyperDX resolves a single trace source on its own. */
const SOURCE = process.env.NEXT_PUBLIC_HYPERDX_SOURCE || '';

export const hyperdxUrl = () => BASE;

/** Half an hour either side of the run. HyperDX scans a time range rather than the whole
 *  table, so the window has to contain the trace -- but a wide one costs a slower query and
 *  risks colliding with a neighbouring run, and investigations take seconds. */
const PAD_MS = 30 * 60 * 1000;

/** A case stores the 32-character OpenTelemetry trace id shared by every span in its run, so
 *  this resolves to the whole investigation rather than one stage of it.
 *
 *  `from` and `to` are not optional in practice. HyperDX only skips its Live Tail default when
 *  the URL carries an explicit range, so a link without one opens on the last few minutes and
 *  finds nothing -- which reads to an operator as "the trace was lost" rather than "you are
 *  looking at the wrong hour". `at` is the time the run happened.
 *
 *  Returns null when the id is absent or the wrong width, for the same reason: no link is
 *  more honest than one that lands on an empty search. */
export function traceUrl(traceId: string, at?: string | Date | null): string | null {
  if (!traceId || !/^[0-9a-f]{32}$/i.test(traceId)) return null;

  const when = at ? new Date(at) : null;
  const centre = when && !Number.isNaN(when.getTime()) ? when.getTime() : Date.now();

  const params = new URLSearchParams({
    traceId,
    from: String(centre - PAD_MS),
    to: String(centre + PAD_MS),
  });
  if (SOURCE) params.set('source', SOURCE);

  return `${BASE}/search?${params.toString()}`;
}
