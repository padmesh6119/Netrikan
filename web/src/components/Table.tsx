import clsx from 'clsx'
import { ArrowDown, ArrowUp } from '@phosphor-icons/react'
import { useMemo, useState, type ReactNode } from 'react'

export interface Column<R> {
  key: string
  label: ReactNode
  value?: (r: R) => number | string | null | undefined
  render?: (r: R) => ReactNode
  align?: 'left' | 'right'
  sortable?: boolean
  width?: string
  title?: string
}

export function Table<R>({ rows, columns, rowKey, onRowClick, selected, initialSort, maxHeight, dense }: {
  rows: R[]
  columns: Column<R>[]
  rowKey: (r: R, i: number) => string
  onRowClick?: (r: R) => void
  selected?: string | null
  initialSort?: { key: string; dir: 'asc' | 'desc' }
  maxHeight?: number
  dense?: boolean
}) {
  const [sort, setSort] = useState(initialSort)
  const sorted = useMemo(() => {
    if (!sort) return rows
    const c = columns.find((x) => x.key === sort.key)
    if (!c?.value) return rows
    const v = c.value
    return [...rows].sort((a, b) => {
      const x = v(a), y = v(b)
      if (x == null) return 1
      if (y == null) return -1
      const d = x < y ? -1 : x > y ? 1 : 0
      return sort.dir === 'asc' ? d : -d
    })
  }, [rows, columns, sort])

  return (
    <div className="overflow-auto rounded-lg border border-line" style={{ maxHeight }}>
      <table className="w-full border-collapse text-[13px]">
        <thead className="sticky top-0 z-10 bg-surface-2">
          <tr>
            {columns.map((c) => {
              const active = sort?.key === c.key
              const canSort = c.sortable !== false && !!c.value
              return (
                <th
                  key={c.key}
                  title={c.title}
                  style={{ width: c.width }}
                  className={clsx('border-b border-line px-3 py-2 text-[11.5px] font-semibold whitespace-nowrap text-muted', c.align === 'right' ? 'text-right' : 'text-left')}
                >
                  {canSort ? (
                    <button
                      className={clsx('inline-flex items-center gap-1 hover:text-ink', active && 'text-ink', c.align === 'right' && 'flex-row-reverse')}
                      onClick={() => setSort({ key: c.key, dir: active && sort?.dir === 'desc' ? 'asc' : 'desc' })}
                    >
                      {c.label}
                      {active && (sort?.dir === 'desc' ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />)}
                    </button>
                  ) : (
                    c.label
                  )}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => {
            const k = rowKey(r, i)
            return (
              <tr
                key={k}
                onClick={onRowClick ? () => onRowClick(r) : undefined}
                className={clsx(
                  'border-b border-line last:border-0',
                  onRowClick && 'cursor-pointer hover:bg-surface-hover',
                  selected === k && 'bg-accent-wash hover:bg-accent-wash',
                )}
              >
                {columns.map((c) => (
                  <td key={c.key} className={clsx('px-3 tnum', dense ? 'py-1.5' : 'py-2', c.align === 'right' ? 'text-right' : 'text-left')}>
                    {c.render ? c.render(r) : (c.value?.(r) ?? '-')}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Inline magnitude bar for 0..1 scores inside a table cell, with an optional reference tick (e.g. 0.5 = chance). */
export function ScoreBar({ value, reference, best, format }: { value: number | null | undefined; reference?: number; best?: boolean; format: (v: number) => string }) {
  if (value == null) return <span className="text-muted">-</span>
  return (
    <div className="flex items-center justify-end gap-2">
      <span className={clsx('w-9 text-right', best ? 'font-semibold text-ink' : 'text-ink-2')}>{format(value)}</span>
      <div className="relative h-1.5 w-16 shrink-0 rounded-full bg-surface-2">
        <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: best ? 'var(--accent)' : 'var(--neutral-series)' }} />
        {reference != null && <div className="absolute -inset-y-0.5 w-px bg-ink-2" style={{ left: `${reference * 100}%` }} />}
      </div>
    </div>
  )
}
