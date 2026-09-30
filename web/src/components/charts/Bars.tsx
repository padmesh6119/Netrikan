import clsx from 'clsx'
import { signed } from '../../lib/format'

/** Horizontal probability bars (0..1) with a value at the tip; one colour per entity. */
export function ProbBars({ items, highlight }: { items: { label: string; value: number; color: string; sub?: string }[]; highlight?: string }) {
  return (
    <div className="space-y-2.5">
      {items.map((it) => (
        <div key={it.label} className="grid grid-cols-[minmax(0,11rem)_1fr_3rem] items-center gap-3">
          <div className="min-w-0">
            <div className={clsx('truncate text-[13px]', highlight === it.label ? 'font-semibold text-ink' : 'text-ink-2')}>{it.label}</div>
            {it.sub && <div className="truncate text-[11px] text-muted">{it.sub}</div>}
          </div>
          <div className="h-2.5 rounded-full bg-surface-2">
            <div className="h-full rounded-full" style={{ width: `${Math.max(0.5, it.value * 100)}%`, background: it.color }} />
          </div>
          <div className="text-right text-[13px] font-medium tnum">{(it.value * 100).toFixed(it.value < 0.1 && it.value > 0 ? 1 : 0)}%</div>
        </div>
      ))}
    </div>
  )
}

/** TreeSHAP contributions in log-odds: bars grow right (raise risk) or left (lower risk) from a shared zero. */
export function DriverBars({ items }: { items: { feature: string; value: number }[] }) {
  const max = Math.max(1e-9, ...items.map((d) => Math.abs(d.value)))
  return (
    <div>
      <div className="mb-2 grid grid-cols-[minmax(0,1.5fr)_minmax(7rem,1fr)_3.5rem] gap-3 text-[11px] text-muted">
        <span />
        <span className="flex justify-between">
          <span>← lowers risk</span>
          <span>raises risk →</span>
        </span>
        <span className="text-right">log-odds</span>
      </div>
      <div className="space-y-1.5">
        {items.map((d) => {
          const w = (Math.abs(d.value) / max) * 50
          return (
            <div key={d.feature} className="grid grid-cols-[minmax(0,1.5fr)_minmax(7rem,1fr)_3.5rem] items-center gap-3">
              <div className="truncate text-[12.5px] text-ink-2" title={d.feature}>
                {d.feature}
              </div>
              <div className="relative h-4">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
                <div
                  className={clsx('absolute top-0.5 bottom-0.5', d.value >= 0 ? 'rounded-r-[3px]' : 'rounded-l-[3px]')}
                  style={{ left: d.value >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, background: d.value >= 0 ? 'var(--up)' : 'var(--down)' }}
                />
              </div>
              <div className="text-right text-[12.5px] tnum text-ink-2">{signed(d.value)}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
