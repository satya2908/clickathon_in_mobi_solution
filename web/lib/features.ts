import 'server-only';

const ENABLED = new Set(['1', 'true', 'yes', 'on']);

/** Fail closed: absent, false, or malformed values keep model-backed functionality unavailable. */
export function recommendationsEnabled(): boolean {
  return ENABLED.has((process.env.RECOMMENDATIONS_ENABLED ?? '').trim().toLowerCase());
}

/** Whether the console may run `verdict ingest` against a path typed into the browser.
 *
 *  Fails closed for a different reason than the one above: this one runs a program on a path the
 *  client chose, so the default has to be off wherever nobody has deliberately turned it on. */
export function ingestEnabled(): boolean {
  return ENABLED.has((process.env.INGEST_ENABLED ?? '').trim().toLowerCase());
}
