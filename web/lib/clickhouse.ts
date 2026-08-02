import 'server-only';

import { createClient, type ClickHouseClient } from '@clickhouse/client';

/** The same six variables the Python side reads, by the same names, so one `.env` at the
 *  repo root configures the engine and the console together. A second set of names would
 *  guarantee that the console eventually reads a different database than the one that
 *  wrote the cases, and the failure would look like missing data rather than misconfiguration. */
function fromEnv() {
  const host = process.env.CLICKHOUSE_HOST ?? 'localhost';
  const port = process.env.CLICKHOUSE_PORT ?? '8123';
  const secure = (process.env.CLICKHOUSE_SECURE ?? 'false').toLowerCase() === 'true';
  return {
    url: `${secure ? 'https' : 'http'}://${host}:${port}`,
    username: process.env.CLICKHOUSE_USER ?? 'default',
    password: process.env.CLICKHOUSE_PASSWORD ?? '',
    database: process.env.CLICKHOUSE_DATABASE ?? 'verdict',
  };
}

/** Next dev recompiles modules on every edit, and a fresh pool per recompile exhausts the
 *  connection limit on a cloud service within a few saves. */
declare global {
  var __verdictCh: ClickHouseClient | undefined;
  var __verdictChWrite: ClickHouseClient | undefined;
}

export function ch(): ClickHouseClient {
  if (!globalThis.__verdictCh) {
    globalThis.__verdictCh = createClient({
      ...fromEnv(),
      clickhouse_settings: {
        // The console is read-only by construction. A dashboard cannot be the thing that
        // mutates the evidence it is displaying.
        readonly: '1',
        max_execution_time: 30,
      },
    });
  }
  return globalThis.__verdictCh;
}

/** Every read goes through here. Query failures are logged with context and rethrown so the
 *  page's error boundary can distinguish an outage from a valid empty result. Returning `[]`
 *  here would turn broken SQL or an unreachable database into "no incidents". */
export async function rows<T>(query: string, params: Record<string, unknown> = {}): Promise<T[]> {
  try {
    const result = await ch().query({
      query,
      query_params: params,
      format: 'JSONEachRow',
    });
    return await result.json<T>();
  } catch (err) {
    console.error(`[clickhouse] ${(err as Error).message}\n${query.slice(0, 400)}`);
    throw err;
  }
}

/** A second, writable client, used for exactly one table.
 *
 *  The read client above is `readonly: '1'` on purpose, and that stays true: a console must
 *  not be able to alter the evidence it displays. Recommendations are the one thing the
 *  console produces rather than reports -- generating them costs about two minutes of model
 *  time, so they are cached rather than recomputed on every page view, and caching means
 *  writing. Keeping it in a separate client means the read path cannot acquire write
 *  permission by accident, and a stray mutation in a query would still be refused. */
export function chWrite(): ClickHouseClient {
  if (!globalThis.__verdictChWrite) {
    globalThis.__verdictChWrite = createClient({
      ...fromEnv(),
      clickhouse_settings: { max_execution_time: 30 },
    });
  }
  return globalThis.__verdictChWrite;
}

/** ClickHouse `DateTime` comes back as `YYYY-MM-DD hh:mm:ss` with no zone marker, and the
 *  columns are all `DateTime('UTC')`. Left alone, `new Date(...)` on that string reads it as
 *  local time -- the same class of bug that shifted every analysis window by 5.5 hours on
 *  the Python side before `as_utc` was added. */
export const iso = (dt: string) => (dt ? `${dt.replace(' ', 'T')}Z` : '');
