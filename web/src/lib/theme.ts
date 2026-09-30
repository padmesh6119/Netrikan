import { useEffect, useState } from 'react'

/** The theme actually on screen: the explicit choice (data-theme on <html>) or, for "system", the OS setting. */
export function useResolvedTheme(): 'light' | 'dark' {
  const read = () => {
    const t = document.documentElement.dataset.theme
    if (t === 'light' || t === 'dark') return t
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  const [theme, setTheme] = useState<'light' | 'dark'>(read)
  useEffect(() => {
    const update = () => setTheme(read())
    const mo = new MutationObserver(update)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', update)
    return () => {
      mo.disconnect()
      mq.removeEventListener('change', update)
    }
  }, [])
  return theme
}
