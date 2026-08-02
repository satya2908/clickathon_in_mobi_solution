import 'server-only';

const ENABLED = new Set(['1', 'true', 'yes', 'on']);

/** Fail closed: absent, false, or malformed values keep model-backed functionality unavailable. */
export function recommendationsEnabled(): boolean {
  return ENABLED.has((process.env.RECOMMENDATIONS_ENABLED ?? '').trim().toLowerCase());
}
