import { useCallback, useEffect, useState } from 'react'

/** Per-viewer state kept in this browser only (triage status, checklists). Reads/writes never throw. */
export function useLocalState<T>(key: string, initial: T) {
  const read = useCallback((): T => {
    try {
      const v = localStorage.getItem(key)
      return v ? (JSON.parse(v) as T) : initial
    } catch {
      return initial
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  const [v, setV] = useState<T>(read)
  useEffect(() => setV(read()), [read])
  const set = useCallback(
    (next: T | ((p: T) => T)) =>
      setV((prev) => {
        const n = typeof next === 'function' ? (next as (p: T) => T)(prev) : next
        try {
          localStorage.setItem(key, JSON.stringify(n))
        } catch {
          /* storage unavailable: keep in memory */
        }
        return n
      }),
    [key],
  )
  return [v, set] as const
}
