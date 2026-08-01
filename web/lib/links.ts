/** Where the trace UI lives. Two deployments are supported and they need different URLs:
 *  the hosted HyperDX that ships with ClickHouse Cloud, and a self-hosted ClickStack
 *  container on 8080. `NEXT_PUBLIC_` because the link is clicked in the browser.
 *
 *  Defaults to the hosted one, which is what runs today -- the `clickstack` service exists
 *  in the compose file but sits behind the `selfhosted` profile and is not started. */
const BASE = process.env.NEXT_PUBLIC_HYPERDX_URL || 'https://hyperdx.clickhouse.cloud';

export const hyperdxUrl = () => BASE;

/** A case stores the 32-character OpenTelemetry trace id shared by every span in its run,
 *  so this resolves to the whole investigation rather than one stage of it. Returns null
 *  when the id is absent or is the wrong width -- a link that lands on an empty search is
 *  worse than no link, because it reads as "the trace is gone" rather than "never recorded". */
export function traceUrl(traceId: string): string | null {
  if (!traceId || traceId.length !== 32) return null;
  return `${BASE}/search?q=${encodeURIComponent(`trace_id:"${traceId}"`)}`;
}
