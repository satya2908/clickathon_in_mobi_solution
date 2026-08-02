import { Console } from '@/components/Console';
import { ingestEnabled, recommendationsEnabled } from '@/lib/features';
import { getDashboard } from '@/lib/queries';

/** Read on every request. A root-cause console that serves a cached page is showing you
 *  the state of the world at build time, which for this product is the one thing it must
 *  never do. */
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export default async function Page() {
  const { run, runs, cases, series, spans, coverageGaps, empty } = await getDashboard();

  return (
    <Console
      run={run}
      runs={runs}
      cases={cases}
      series={series}
      spans={spans}
      coverageGaps={coverageGaps}
      recommendationsEnabled={recommendationsEnabled()}
      ingestEnabled={ingestEnabled()}
      empty={empty}
    />
  );
}
