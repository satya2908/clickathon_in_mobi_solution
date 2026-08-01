/** Drawn inline rather than pulled from an icon package, following the reference
 *  console: `currentColor` lets them follow the theme for free. Five shapes, all of
 *  which carry meaning — none are decoration beside a label. */

const base = {
  width: 14,
  height: 14,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.4,
  strokeLinejoin: 'round' as const,
  strokeLinecap: 'round' as const,
};

export function ChevronIcon({ size = 11 }: { size?: number }) {
  return (
    <svg {...base} width={size} height={size} aria-hidden="true">
      <path d="M6 3.4 10.6 8 6 12.6" />
    </svg>
  );
}

export function SearchIcon() {
  return (
    <svg {...base} width={12} height={12} aria-hidden="true">
      <circle cx="7.2" cy="7.2" r="4.3" />
      <path d="m10.4 10.4 2.7 2.7" />
    </svg>
  );
}

export function AlertIcon({ size = 13 }: { size?: number }) {
  return (
    <svg {...base} width={size} height={size} aria-hidden="true">
      <path d="M8 2.6 14.2 13H1.8L8 2.6Z" />
      <path d="M8 6.6v3" />
      <circle cx="8" cy="11.2" r="0.55" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function CloseIcon() {
  return (
    <svg {...base} width={12} height={12} aria-hidden="true">
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

export function LinkIcon() {
  return (
    <svg {...base} width={11} height={11} aria-hidden="true">
      <path d="M6.6 9.4a2.6 2.6 0 0 0 3.9.3l2-2a2.6 2.6 0 0 0-3.7-3.7l-1.1 1.1" />
      <path d="M9.4 6.6a2.6 2.6 0 0 0-3.9-.3l-2 2a2.6 2.6 0 0 0 3.7 3.7l1.1-1.1" />
    </svg>
  );
}

/** Points at the active sort direction; faded when the column is not the sort key. */
export function SortIcon({ on }: { on: boolean }) {
  return (
    <svg {...base} width={9} height={9} aria-hidden="true" style={{ opacity: on ? 1 : 0.32 }}>
      <path d="M8 3.6v8.8M4.8 9.2 8 12.4l3.2-3.2" />
    </svg>
  );
}
