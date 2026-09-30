import clsx from 'clsx'
import { Info, MagnifyingGlass, Warning } from '@phosphor-icons/react'
import type { ReactNode } from 'react'

export function Card({ title, subtitle, actions, children, className, pad = true }: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  pad?: boolean
}) {
  return (
    <section className={clsx('rounded-xl border border-line bg-surface', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 px-5 pt-4">
          <div className="min-w-0">
            {title && <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-[13px] text-ink-2">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(pad && 'p-5', pad && (title || actions) && 'pt-3')}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: 'critical' | 'good' }) {
  return (
    <div className="min-w-0 rounded-xl border border-line bg-surface px-4 py-3.5">
      <div className="truncate text-[12px] font-medium text-muted">{label}</div>
      <div className={clsx('mt-1 text-[22px] font-semibold leading-tight tracking-tight', tone === 'critical' && 'text-critical', tone === 'good' && 'text-good-ink')}>
        {value}
      </div>
      {hint && <div className="mt-1 text-[12px] leading-snug text-ink-2">{hint}</div>}
    </div>
  )
}

export function StatRow({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">{children}</div>
}

export function Tabs<T extends string>({ value, onChange, items }: { value: T; onChange: (v: T) => void; items: { id: T; label: string; count?: ReactNode }[] }) {
  return (
    <div role="tablist" className="flex gap-1 overflow-x-auto border-b border-line [scrollbar-width:none]">
      {items.map((it) => (
        <button
          key={it.id}
          role="tab"
          aria-selected={value === it.id}
          onClick={() => onChange(it.id)}
          className={clsx(
            '-mb-px flex shrink-0 items-center gap-2 border-b-2 px-3 py-2.5 text-[13.5px] font-medium transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-accent',
            value === it.id ? 'border-accent text-ink' : 'border-transparent text-muted hover:text-ink',
          )}
        >
          {it.label}
          {it.count != null && <span className="rounded-full bg-surface-2 px-1.5 text-[11px] text-ink-2 tnum">{it.count}</span>}
        </button>
      ))}
    </div>
  )
}

export function Segmented<T extends string | number>({ value, onChange, items, size = 'md' }: {
  value: T
  onChange: (v: T) => void
  items: { id: T; label: ReactNode; title?: string }[]
  size?: 'sm' | 'md'
}) {
  return (
    <div className="inline-flex w-fit rounded-lg border border-line bg-surface-2 p-0.5">
      {items.map((it) => (
        <button
          key={String(it.id)}
          title={it.title}
          onClick={() => onChange(it.id)}
          className={clsx(
            'rounded-md font-medium whitespace-nowrap transition-colors duration-200 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-accent',
            size === 'sm' ? 'px-2 py-1 text-[12px]' : 'px-3 py-1.5 text-[13px]',
            value === it.id ? 'bg-surface text-ink shadow-sm ring-1 ring-line' : 'text-ink-2 hover:text-ink',
          )}
        >
          {it.label}
        </button>
      ))}
    </div>
  )
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="flex min-w-0 flex-col gap-1.5">
      <span className="text-[12.5px] font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="text-[12px] text-muted">{hint}</span>}
    </label>
  )
}

export function Select<T extends string>({ value, onChange, options, className }: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
  className?: string
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className={clsx(
        'h-9 min-w-0 rounded-lg border border-line bg-surface pr-8 pl-3 text-[13.5px] text-ink outline-none hover:border-line-strong focus-visible:ring-2 focus-visible:ring-accent',
        className,
      )}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function Slider({ value, onChange, min, max, step, format }: {
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  format: (v: number) => string
}) {
  return (
    <div className="flex h-9 items-center gap-3">
      <input type="range" className="w-full min-w-24" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="w-12 shrink-0 text-right text-[13px] font-medium tnum">{format(value)}</span>
    </div>
  )
}

export function Toolbar({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-1 gap-4 rounded-xl border border-line bg-surface p-4 sm:grid-cols-2 lg:flex lg:flex-wrap lg:items-start lg:gap-6">{children}</div>
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'critical' | 'good' | 'accent' | 'warning' }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-[5px] px-1.5 py-0.5 text-[11.5px] font-medium whitespace-nowrap',
        tone === 'neutral' && 'bg-surface-2 text-ink-2',
        tone === 'critical' && 'bg-critical/12 text-critical',
        tone === 'good' && 'bg-good/12 text-good-ink',
        tone === 'accent' && 'bg-accent-wash text-accent-ink',
        tone === 'warning' && 'bg-warning/18 text-ink',
      )}
    >
      {children}
    </span>
  )
}

export function Swatch({ color, shape = 'square' }: { color: string; shape?: 'square' | 'line' | 'dot' | 'dash' }) {
  if (shape === 'line') return <span className="inline-block h-0.5 w-3.5 rounded" style={{ background: color }} />
  if (shape === 'dash') return <span className="inline-block w-3.5 border-t-2 border-dashed" style={{ borderColor: color }} />
  return <span className={clsx('inline-block size-2.5', shape === 'dot' ? 'rounded-full' : 'rounded-[3px]')} style={{ background: color }} />
}

export function Legend({ items }: { items: { label: string; color: string; shape?: 'square' | 'line' | 'dot' | 'dash' }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-ink-2">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          <Swatch color={it.color} shape={it.shape} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

export function Note({ children, tone = 'info', title }: { children: ReactNode; tone?: 'info' | 'caveat'; title?: string }) {
  const Icon = tone === 'caveat' ? Warning : Info
  return (
    <div className={clsx('flex gap-3 rounded-lg border px-4 py-3 text-[13px] leading-relaxed', tone === 'caveat' ? 'border-warning/40 bg-warning/8' : 'border-line bg-surface-2')}>
      <Icon className={clsx('mt-0.5 size-4 shrink-0', tone === 'caveat' ? 'text-warning' : 'text-muted')} />
      <div className="min-w-0 text-ink-2">
        {title && <div className="mb-0.5 font-semibold text-ink">{title}</div>}
        {children}
      </div>
    </div>
  )
}

export function Findings({ items }: { items: { title: string; body: ReactNode }[] }) {
  return (
    <ol className="space-y-3">
      {items.map((f, i) => (
        <li key={f.title} className="flex gap-3">
          <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-surface-2 text-[11px] font-semibold text-ink-2 tnum">{i + 1}</span>
          <div className="text-[13.5px] leading-relaxed text-ink-2">
            <span className="font-semibold text-ink">{f.title}</span> {f.body}
          </div>
        </li>
      ))}
    </ol>
  )
}

/** Skeleton shaped like a chart card, with a line saying what the server is computing. */
export function Loading({ label = 'Loading', className, rows = 3 }: { label?: string; className?: string; rows?: number }) {
  return (
    <div className={clsx('space-y-3 py-2', className)} role="status" aria-live="polite">
      <div className="skeleton h-4 w-40" />
      <div className="skeleton h-40 w-full" />
      {Array.from({ length: rows - 1 }, (_, i) => (
        <div key={i} className="skeleton h-3" style={{ width: `${72 - i * 18}%` }} />
      ))}
      <p className="pt-1 text-[12.5px] text-muted">{label}...</p>
    </div>
  )
}

export function ErrorState({ error }: { error: Error | null }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-critical/30 bg-critical/6 px-4 py-3 text-[13px]">
      <Warning className="mt-0.5 size-4 shrink-0 text-critical" />
      <div>
        <div className="font-semibold">This view could not load</div>
        <div className="text-ink-2">{error?.message ?? 'Unknown error'}</div>
      </div>
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-14 text-center text-[13px] text-muted">
      <MagnifyingGlass className="size-5" />
      {children}
    </div>
  )
}

/** Shows a subtle progress bar while a query refetches with the previous data still on screen. */
export function Busy({ on }: { on: boolean }) {
  return (
    <div className={clsx('pointer-events-none fixed inset-x-0 top-0 z-50 h-0.5 overflow-hidden transition-opacity', on ? 'opacity-100' : 'opacity-0')}>
      <div className="h-full w-1/3 animate-[busy_1.1s_ease-in-out_infinite] bg-accent" />
      <style>{'@keyframes busy{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}'}</style>
    </div>
  )
}

export function PageHeader({ title, children, right }: { title: string; children?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 max-w-3xl">
        <h1 className="text-[26px] leading-tight font-semibold tracking-[-0.02em]">{title}</h1>
        {children && <p className="mt-2 max-w-[70ch] text-[14px] leading-relaxed text-ink-2">{children}</p>}
      </div>
      {right}
    </div>
  )
}
