import { useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'

/** Page state lives in the URL, so every view (day, host, minute, tab) is shareable and survives reload. */
export function useUrlState() {
  const [sp, setSp] = useSearchParams()
  const get = useCallback((k: string, fallback = '') => sp.get(k) ?? fallback, [sp])
  const set = useCallback(
    (patch: Record<string, string | number | null | undefined>) =>
      setSp(
        (prev) => {
          const n = new URLSearchParams(prev)
          for (const [k, v] of Object.entries(patch)) {
            if (v === null || v === undefined || v === '') n.delete(k)
            else n.set(k, String(v))
          }
          return n
        },
        { replace: true },
      ),
    [setSp],
  )
  return { get, set }
}
