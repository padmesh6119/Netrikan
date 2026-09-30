import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'

/** Width of a container, tracked with ResizeObserver so SVG charts stay crisp at any size. */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [w, setW] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)))
    ro.observe(el)
    setW(Math.floor(el.getBoundingClientRect().width))
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

/** Tooltip positioned inside the chart box, flipped to stay within it. */
export function Tip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const left = x > width - 220 ? x - 14 : x + 14
  return (
    <div
      className="pointer-events-none absolute z-20 min-w-36 rounded-lg border border-line bg-surface px-3 py-2 text-[12px] shadow-lg shadow-black/10"
      style={{ left, top: Math.max(0, y - 10), transform: x > width - 220 ? 'translateX(-100%)' : undefined }}
    >
      {children}
    </div>
  )
}

export function TipRow({ color, label, value, shape = 'dot' }: { color?: string; label: ReactNode; value: ReactNode; shape?: 'dot' | 'line' }) {
  return (
    <div className="flex items-center justify-between gap-4 py-0.5">
      <span className="flex items-center gap-1.5 text-ink-2">
        {color && (shape === 'line' ? <span className="h-0.5 w-3 rounded" style={{ background: color }} /> : <span className="size-2 rounded-full" style={{ background: color }} />)}
        {label}
      </span>
      <span className="font-semibold text-ink tnum">{value}</span>
    </div>
  )
}

export const AXIS_TEXT = { fontSize: 11, fill: 'var(--muted)' } as const
