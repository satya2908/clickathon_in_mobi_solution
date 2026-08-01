'use client';

import { useEffect, useRef, useState } from 'react';

/** LibreChat, embedded, holding the ClickHouse MCP server.
 *
 *  Three things worth knowing about this integration.
 *
 *  The iframe is created on first open, never on load. LibreChat is a full application, and
 *  booting one inside every page view to serve the readers who never click would make the
 *  console slower for everyone to benefit a few. Once created it is kept, so reopening does
 *  not throw away the conversation.
 *
 *  There is no cross-frame messaging. Handing the open case into the conversation would need
 *  postMessage on both sides and a LibreChat that listens for it, which it does not. The
 *  model reaches the same warehouse the page was rendered from, so a typed question about
 *  `os_version=Android 15` finds the same rows the table is showing.
 *
 *  The host page must be served over http, not opened from the filesystem. LibreChat keeps a
 *  session cookie, and a `file://` page makes that cookie third-party, which Chrome drops --
 *  the symptom is a login screen that reappears on every reload and reads as a LibreChat bug
 *  rather than an embedding mistake. */
const CHAT_URL = process.env.NEXT_PUBLIC_CHAT_URL || 'http://localhost:3080';

export function VerdictAI() {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      // The case panel also closes on Escape. It is a modal dialog above this one, so it
      // takes the key first and this only fires when the chat is the top surface.
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  const toggle = () => {
    setMounted(true);
    setOpen(o => !o);
  };

  return (
    <>
      <div className={`vai-panel${open ? ' open' : ''}`} ref={panel} role="dialog" aria-label="Verdict.AI" aria-hidden={!open}>
        <div className="vai-head">
          <span className="vai-mark">V</span>
          <span className="vai-title">Verdict.AI</span>
          <span className="vai-sub">reads the same warehouse</span>
          <button className="vai-x" onClick={() => setOpen(false)} aria-label="Close Verdict.AI">
            ×
          </button>
        </div>
        {mounted && <iframe src={CHAT_URL} title="Verdict.AI" allow="clipboard-write" />}
      </div>

      <button className={`vai-toggle${open ? ' on' : ''}`} onClick={toggle} aria-expanded={open} title="Ask Verdict.AI (⌘/)">
        <span className="vai-mark sm">V</span>
        {open ? 'Close' : 'Verdict.AI'}
      </button>
    </>
  );
}
