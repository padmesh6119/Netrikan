import { bisector } from 'd3-array'
import { scaleLinear, scaleUtc } from 'd3-scale'
import { line as d3line } from 'd3-shape'
import { motion } from 'motion/react'
import { useMemo, useState, type ReactNode } from 'react'
import { hhmm, stamp, tick } from '../../lib/format'
import { AXIS_TEXT, Tip, TipRow, useWidth } from './core'

export interface Series {
  id: string
  label: string
  color: string
  t: number[]
  v: number[]
  dashed?: boolean
}
export interface Band {
  start: number
  end: number
  color: string
  label: string
}

const M = { top: 12, right: 16, bottom: 26, left: 44 }
const bis = bisector((d: number) => d).center

/** Merge consecutive 1-minute category cells into spans so a long attack draws as one band. */
export function toBands(t: number[], cat: (number | string | null)[], color: (c: number | string) => string, label: (c: number | string) => string, step = 60_000): Band[] {
  const out: Band[] = []
  for (let i = 0; i < t.length; i++) {
    const c = cat[i]
    if (c === null || c === -1 || c === 0 || c === '') continue
    const last = out[out.length - 1]
    if (last && last.label === label(c) && t[i] - last.end < step / 2 + 1) last.end = t[i] + step
    else out.push({ start: t[i], end: t[i] + step, color: color(c), label: label(c) })
  }
  return out
}

export function TimeChart({ series, bands = [], rules = [], points = [], threshold, yMax, yLabel, yFormat = (v) => v.toFixed(2), height = 300, selected, onPick, extraTip }: {
  series: Series[]
  bands?: Band[]
  rules?: { t: number; color: string }[]
  points?: { t: number; v: number; color: string }[]
  threshold?: { v: number; label: string }
  yMax?: number
  yLabel?: string
  yFormat?: (v: number) => string
  height?: number
  selected?: number | null
  onPick?: (t: number, i: number) => void
  extraTip?: (t: number) => ReactNode
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ i: number; px: number; py: number } | null>(null)
  const base = series[0]
  const iw = Math.max(0, width - M.left - M.right)
  const ih = height - M.top - M.bottom

  const { x, y } = useMemo(() => {
    const ts = series.flatMap((s) => [s.t[0], s.t[s.t.length - 1]]).filter((v) => v != null)
    const lo = Math.min(...ts), hi = Math.max(...ts) + 60_000
    const vmax = yMax ?? Math.max(0.05, ...series.flatMap((s) => s.v), threshold?.v ?? 0) * 1.12
    return { x: scaleUtc().domain([lo, hi]).range([0, iw]), y: scaleLinear().domain([0, vmax]).range([ih, 0]).nice() }
  }, [series, iw, ih, yMax, threshold])

  const paths = useMemo(() => {
    return series.map(
      (s) =>
        d3line<number>()
          .x((_, i) => x(s.t[i] + 30_000))
          .y((v) => y(v))
          .defined((v, i) => v != null && !Number.isNaN(v) && (i === 0 || s.t[i] - s.t[i - 1] <= 60_000))(s.v) ?? '',
    )
  }, [series, x, y])

  const drawKey = base ? `${base.t[0]}-${base.t.length}-${base.v[0]}-${base.v[base.v.length - 1]}` : ''
  if (!base || !base.t.length) return <div ref={ref} style={{ height }} />
  const xt = x.ticks(Math.max(2, Math.floor(iw / 110)))
  const fmt = tick
  const yt = y.ticks(4)

  const move = (e: React.MouseEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect()
    const px = e.clientX - r.left
    const i = bis(base.t, x.invert(px).getTime() - 30_000)
    setHover({ i, px: x(base.t[i] + 30_000) + M.left, py: e.clientY - r.top + M.top })
  }
  const hv = hover ? base.t[hover.i] : null

  return (
    <div ref={ref} className="relative select-none" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} className="block overflow-visible" role="img" aria-label={yLabel}>
          <g transform={`translate(${M.left},${M.top})`}>
            <motion.g key={'bands' + drawKey} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
              {bands.map((b, k) => (
                <rect key={k} x={x(b.start)} y={0} width={Math.max(1.5, x(b.end) - x(b.start))} height={ih} style={{ fill: b.color }} opacity={0.3} />
              ))}
            </motion.g>
            {yt.map((v) => (
              <g key={v} transform={`translate(0,${y(v)})`}>
                <line x2={iw} style={{ stroke: v === 0 ? 'var(--axis)' : 'var(--grid)' }} />
                <text x={-8} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">
                  {yFormat(v)}
                </text>
              </g>
            ))}
            {xt.map((t) => (
              <text key={+t} x={x(t)} y={ih + 18} textAnchor="middle" style={AXIS_TEXT} className="tnum">
                {fmt(t)}
              </text>
            ))}
            {rules.map((r, k) => (
              <line key={k} x1={x(r.t + 30_000)} x2={x(r.t + 30_000)} y1={0} y2={ih} style={{ stroke: r.color }} strokeWidth={1.25} opacity={0.75} />
            ))}
            {series.map((s, k) =>
              s.dashed ? (
                <motion.path key={s.id + drawKey} d={paths[k]} fill="none" style={{ stroke: s.color }} strokeWidth={1.5} strokeDasharray="4 3" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }} />
              ) : (
                // Lines draw themselves in left to right whenever the data changes (new host, day or model).
                <motion.path key={s.id + drawKey} d={paths[k]} fill="none" style={{ stroke: s.color }} strokeWidth={k === 0 ? 2 : 1.5} strokeLinejoin="round" strokeLinecap="round"
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }} />
              ),
            )}
            <motion.g key={'pts' + drawKey} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4, delay: 0.7 }}>
              {points.map((p, k) => (
                <circle key={k} cx={x(p.t + 30_000)} cy={y(p.v)} r={4} style={{ fill: p.color, stroke: 'var(--surface)' }} strokeWidth={2} />
              ))}
            </motion.g>
            {threshold && (
              <g transform={`translate(0,${y(threshold.v)})`}>
                <line x2={iw} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="5 4" strokeWidth={1} />
                <text x={4} y={-5} textAnchor="start" style={{ ...AXIS_TEXT, fill: 'var(--ink-2)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}>
                  {threshold.label}
                </text>
              </g>
            )}
            {selected != null && (
              <g transform={`translate(${x(selected + 30_000)},0)`}>
                <line y2={ih} style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />
                <path d="M-5,-6 L5,-6 L0,0 Z" style={{ fill: 'var(--ink)' }} />
              </g>
            )}
            {hv != null && (
              <g>
                <line x1={x(hv + 30_000)} x2={x(hv + 30_000)} y2={ih} style={{ stroke: 'var(--ink-2)' }} strokeWidth={1} opacity={0.5} />
                {series.map((s) => {
                  const j = s === base ? hover!.i : bis(s.t, hv)
                  const v = s.v[j]
                  return v == null ? null : <circle key={s.id} cx={x(s.t[j] + 30_000)} cy={y(v)} r={4} style={{ fill: s.color, stroke: 'var(--surface)' }} strokeWidth={2} />
                })}
              </g>
            )}
            <rect
              width={iw}
              height={ih}
              fill="transparent"
              className={onPick ? 'cursor-crosshair' : undefined}
              onMouseMove={move}
              onMouseLeave={() => setHover(null)}
              onClick={() => hover && onPick?.(base.t[hover.i], hover.i)}
            />
          </g>
        </svg>
      )}
      {hover && hv != null && (
        <Tip x={hover.px} y={hover.py} width={width}>
          <div className="mb-1 font-semibold text-ink">{stamp(hv)}</div>
          {series.map((s) => {
            const j = s === base ? hover.i : bis(s.t, hv)
            return <TipRow key={s.id} color={s.color} shape="line" label={s.label} value={s.v[j] == null ? '-' : yFormat(s.v[j])} />
          })}
          {bands.filter((b) => hv >= b.start && hv < b.end).map((b) => (
            <TipRow key={b.label} color={b.color} label={b.label} value={`${hhmm(b.start)}-${hhmm(b.end)}`} />
          ))}
          {extraTip?.(hv)}
          {onPick && <div className="mt-1 border-t border-line pt-1 text-[11px] text-muted">Click to inspect this minute</div>}
        </Tip>
      )}
    </div>
  )
}
