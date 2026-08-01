'use client';

import { useEffect, useState } from 'react';

type Theme = 'dark' | 'light';

const STORAGE_KEY = 'verdict-theme';

export function ThemeToggle() {
  // Null until mounted. The inline bootstrap has already resolved the theme, so
  // adopting whatever it chose is safer than assuming a default here and writing
  // that assumption straight back over it.
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY) as Theme | null;
    if (stored === 'dark' || stored === 'light') setTheme(stored);
    else setTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  }, []);

  useEffect(() => {
    if (!theme) return;
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const dark = theme === 'dark';

  return (
    <button
      className="btn sm"
      onClick={() => setTheme(dark ? 'light' : 'dark')}
      title={`Switch to ${dark ? 'light' : 'dark'} theme`}
      aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`}
    >
      {dark ? '\u263e' : '\u2600'}
    </button>
  );
}
