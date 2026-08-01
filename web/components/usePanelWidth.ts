'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const KEY = 'verdict-panel-width';
const DEFAULT_FRACTION = 0.62;
const MIN = 560;
// Always leave enough of the case list showing to pick a different row without closing the
// panel first. A drawer covering the whole window is a page, and a page should have a URL.
const EDGE = 260;

const clamp = (px: number, viewport: number) => Math.min(Math.max(px, MIN), Math.max(MIN, viewport - EDGE));

/** A width for the case panel that the reader controls and the browser remembers.
 *
 *  The panel was a fixed `min(1180px, 94vw)`, which is two different mistakes on two
 *  different monitors: cramped next to a wide trace, and covering the whole case list on a
 *  laptop. How much of the list somebody wants to keep in view while reading a case is not a
 *  thing this component can know, so it stops guessing.
 *
 *  Stored as a fraction of the viewport rather than pixels, so the choice survives moving the
 *  window to a different display instead of leaving a 1600px panel on a 1280px screen. */
export function usePanelWidth() {
  const [width, setWidth] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const frame = useRef(0);

  // Read after mount: the server has no viewport, and rendering a guessed width first would
  // show the panel jumping to its real size on hydration.
  useEffect(() => {
    const stored = Number(localStorage.getItem(KEY));
    const fraction = stored > 0.2 && stored < 0.95 ? stored : DEFAULT_FRACTION;
    setWidth(clamp(window.innerWidth * fraction, window.innerWidth));
  }, []);

  useEffect(() => {
    const onResize = () => setWidth(w => (w === null ? w : clamp(w, window.innerWidth)));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const commit = useCallback((px: number) => {
    const next = clamp(px, window.innerWidth);
    setWidth(next);
    localStorage.setItem(KEY, String(next / window.innerWidth));
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      const handle = e.currentTarget;
      handle.setPointerCapture(e.pointerId);
      setDragging(true);

      const move = (ev: PointerEvent) => {
        // One update per frame. Pointer events fire faster than the panel can lay out, and
        // the panel contains a trace tree of several hundred rows.
        cancelAnimationFrame(frame.current);
        frame.current = requestAnimationFrame(() => setWidth(clamp(window.innerWidth - ev.clientX, window.innerWidth)));
      };
      const up = () => {
        cancelAnimationFrame(frame.current);
        setDragging(false);
        handle.removeEventListener('pointermove', move);
        handle.removeEventListener('pointerup', up);
        handle.removeEventListener('pointercancel', up);
        setWidth(w => {
          if (w !== null) localStorage.setItem(KEY, String(w / window.innerWidth));
          return w;
        });
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener('pointerup', up);
      handle.addEventListener('pointercancel', up);
    },
    [],
  );

  // The handle is a real separator: focusable, and movable without a pointer.
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (width === null) return;
      const step = e.shiftKey ? 120 : 24;
      if (e.key === 'ArrowLeft') { e.preventDefault(); commit(width + step); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); commit(width - step); }
      else if (e.key === 'Home') { e.preventDefault(); commit(window.innerWidth); }
      else if (e.key === 'End') { e.preventDefault(); commit(MIN); }
    },
    [width, commit],
  );

  const reset = useCallback(() => commit(window.innerWidth * DEFAULT_FRACTION), [commit]);

  return { width, dragging, onPointerDown, onKeyDown, reset, min: MIN };
}
