import { scaleUtc } from 'd3-scale'
import { useMemo, useState } from 'react'
import { seq } from '../../lib/colors'
import { stamp, tick } from '../../lib/format'
import { AXIS_TEXT, Tip, TipRow, useWidth } from './core'

const ROW = 26
const LABEL_W = 132

/** Hosts x minutes, colour = score (sequential ramp). Optional per-cell truth marker drawn as a strip under the row. */
export function Heatmap({ rows, host, t, v, max = 1, marker, markerColor, markerLabel, valueLabel, format = (x) => x.toFixed(2), onRowClick, selectedRow }: {
  rows: string[]
  host: number[]
  t: number[]
  v: number[]
  max?: number
  marker?: number[]
  markerColor?: (m: number) => string
  markerLabel?: (m: number) => string
  valueLabel: string
  format?: (x: number) => string
  onRowClick?: (row: string) => void
  selectedRow?: string | null
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ k: number; px: number; py: number } | null>(null)
  const iw = Math.max(0, width - LABEL_W - 12)
  const height = rows.length * ROW + 26

  const { x, lookup } = useMemo(() => {
    const lo = Math.min(...t), hi = Math.max(...t) + 60_000
    const lookup = new Map<string, number>()
    t.forEach((tt, k) => lookup.set(`${host[k]}|${tt}`, k))
    return { x: scaleUtc().domain([lo, hi]).range([0, iw]), lookup }
  }, [t, host, iw])
  const cw = Math.max(1, x(60_000 + x.domain()[0].getTime()) - x(x.domain()[0]))

  const move = (e: React.MouseEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left, py = e.clientY - r.top
    const row = Math.floor(py / ROW)
    const tt = Math.floor(x.invert(px).getTime() / 60_000) * 60_000
    const k = lookup.get(`${row}|${tt}`)
    setHover(k == null ? null : { k, px: px + LABEL_W, py })
  }

  return (
    <div ref={ref} className="relative select-none">
      {width > 0 && (
        <svg width={width} height={height} className="block">
          {rows.map((h, r) => (
            <g key={h} transform={`translate(0,${r * ROW})`} className={onRowClick ? 'cursor-pointer' : undefined} onClick={() => onRowClick?.(h)}>
              {selectedRow === h && <rect x={0} width={width} height={ROW} style={{ fill: 'var(--accent-wash)' }} rx={4} />}
              <text x={LABEL_W - 10} y={ROW / 2} dy="0.32em" textAnchor="end" className="tnum" style={{ fontSize: 12, fill: selectedRow === h ? 'var(--ink)' : 'var(--ink-2)', fontFamily: 'var(--font-mono)' }}>
                {h}
              </text>
              <rect x={LABEL_W} y={3} width={iw} height={ROW - 9} style={{ fill: 'var(--seq-0)' }} rx={2} />
            </g>
          ))}
          <g transform={`translate(${LABEL_W},0)`}>
            {t.map((tt, k) => (
              <rect key={k} x={x(tt)} y={host[k] * ROW + 3} width={cw + 0.4} height={ROW - 9} style={{ fill: seq(Math.min(1, v[k] / max)) }} />
            ))}
            {marker &&
              t.map((tt, k) =>
                marker[k] > 0 ? <rect key={`m${k}`} x={x(tt)} y={host[k] * ROW + ROW - 5} width={Math.max(cw, 1.5)} height={3} style={{ fill: markerColor?.(marker[k]) ?? 'var(--critical)' }} /> : null,
              )}
            {x.ticks(Math.max(2, Math.floor(iw / 110))).map((tt) => (
              <text key={+tt} x={x(tt)} y={rows.length * ROW + 16} textAnchor="middle" style={AXIS_TEXT} className="tnum">
                {tick(tt)}
              </text>
            ))}
            <rect width={iw} height={rows.length * ROW} fill="transparent" onMouseMove={move} onMouseLeave={() => setHover(null)}
              onClick={(e) => { const r = Math.floor((e.clientY - e.currentTarget.getBoundingClientRect().top) / ROW); if (rows[r]) onRowClick?.(rows[r]) }}
              className={onRowClick ? 'cursor-pointer' : undefined} />
          </g>
        </svg>
      )}
      {hover && (
        <Tip x={hover.px} y={hover.py} width={width}>
          <div className="mb-1 font-semibold text-ink">{rows[host[hover.k]]}</div>
          <div className="mb-1 text-muted">{stamp(t[hover.k])}</div>
          <TipRow color={seq(Math.min(1, v[hover.k] / max))} label={valueLabel} value={format(v[hover.k])} />
          {marker && marker[hover.k] > 0 && <TipRow color={markerColor?.(marker[hover.k])} label="Ground truth" value={markerLabel?.(marker[hover.k]) ?? 'attack'} />}
        </Tip>
      )}
    </div>
  )
}

export function RampLegend({ label, lo, hi }: { label: string; lo: string; hi: string }) {
  return (
    <div className="flex items-center gap-2 text-[12px] text-ink-2">
      <span>{label}</span>
      <span className="tnum text-muted">{lo}</span>
      <span className="flex h-2.5 overflow-hidden rounded-sm">
        {Array.from({ length: 9 }, (_, i) => (
          <span key={i} className="w-3" style={{ background: `var(--seq-${i})` }} />
        ))}
      </span>
      <span className="tnum text-muted">{hi}</span>
    </div>
  )
}
