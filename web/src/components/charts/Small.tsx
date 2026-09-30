import { scaleLinear, scaleUtc } from 'd3-scale'
import { line as d3line } from 'd3-shape'
import { useState } from 'react'
import { stamp, tick } from '../../lib/format'
import { AXIS_TEXT, Tip, TipRow, useWidth } from './core'

/** One small forecast-vs-observed chart over K minutes ahead. */
export function RolloutChart({ title, forecast, observed }: { title: string; forecast: number[]; observed: (number | null)[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const H = 130, M = { t: 8, r: 10, b: 22, l: 38 }
  const iw = Math.max(0, width - M.l - M.r), ih = H - M.t - M.b
  const K = forecast.length
  const vals = [...forecast, ...observed.filter((v): v is number => v != null)]
  const x = scaleLinear().domain([1, K]).range([0, iw])
  const y = scaleLinear().domain([Math.min(0, ...vals), Math.max(1, ...vals) * 1.1]).range([ih, 0]).nice()
  const path = (arr: (number | null)[]) => d3line<number | null>().defined((v) => v != null).x((_, i) => x(i + 1)).y((v) => y(v ?? 0))(arr) ?? ''
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1))
  return (
    <div className="rounded-lg border border-line p-3">
      <div className="mb-1 truncate text-[12.5px] font-medium text-ink-2">{title}</div>
      <div ref={ref} className="relative" style={{ height: H }}>
        {width > 0 && (
          <svg width={width} height={H}>
            <g transform={`translate(${M.l},${M.t})`}>
              {y.ticks(3).map((v) => (
                <g key={v} transform={`translate(0,${y(v)})`}>
                  <line x2={iw} style={{ stroke: 'var(--grid)' }} />
                  <text x={-6} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{fmt(v)}</text>
                </g>
              ))}
              {Array.from({ length: K }, (_, i) => (
                <text key={i} x={x(i + 1)} y={ih + 16} textAnchor="middle" style={AXIS_TEXT} className="tnum">+{i + 1}m</text>
              ))}
              <path d={path(observed)} fill="none" style={{ stroke: 'var(--ink-2)' }} strokeWidth={1.5} strokeDasharray="4 3" />
              <path d={path(forecast)} fill="none" style={{ stroke: 'var(--s1)' }} strokeWidth={2} strokeLinejoin="round" />
              {forecast.map((v, i) => <circle key={i} cx={x(i + 1)} cy={y(v)} r={3.5} style={{ fill: 'var(--s1)', stroke: 'var(--surface)' }} strokeWidth={1.5} />)}
              {observed.map((v, i) => v == null ? null : <circle key={`o${i}`} cx={x(i + 1)} cy={y(v)} r={3} style={{ fill: 'var(--surface)', stroke: 'var(--ink-2)' }} strokeWidth={1.5} />)}
              {Array.from({ length: K }, (_, i) => (
                <rect key={`h${i}`} x={x(i + 1) - iw / K / 2} width={iw / K} height={ih} fill="transparent" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
              ))}
            </g>
          </svg>
        )}
        {hover != null && (
          <Tip x={M.l + x(hover + 1)} y={20} width={width}>
            <div className="mb-1 font-semibold">+{hover + 1} min</div>
            <TipRow color="var(--s1)" shape="line" label="Simulated" value={fmt(forecast[hover])} />
            <TipRow color="var(--ink-2)" shape="line" label="Observed" value={observed[hover] == null ? '-' : fmt(observed[hover]!)} />
          </Tip>
        )}
      </div>
    </div>
  )
}

/** Column histogram for pre-binned counts. */
export function Histogram({ title, counts, step, color, xLabel }: { title: string; counts: number[]; step: number; color: string; xLabel: string }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const H = 120, M = { t: 6, r: 6, b: 20, l: 30 }
  const iw = Math.max(0, width - M.l - M.r), ih = H - M.t - M.b
  const n = counts.length
  const x = scaleLinear().domain([0, n * step]).range([0, iw])
  const y = scaleLinear().domain([0, Math.max(1, ...counts)]).range([ih, 0]).nice()
  const bw = Math.max(1, iw / n - 2)
  return (
    <div className="rounded-lg border border-line p-3">
      <div className="mb-1 flex items-center gap-1.5 text-[12.5px] font-medium text-ink-2">
        <span className="size-2.5 rounded-[3px]" style={{ background: color }} />
        {title}
      </div>
      <div ref={ref} className="relative" style={{ height: H }}>
        {width > 0 && (
          <svg width={width} height={H}>
            <g transform={`translate(${M.l},${M.t})`}>
              {y.ticks(3).map((v) => (
                <g key={v} transform={`translate(0,${y(v)})`}>
                  <line x2={iw} style={{ stroke: v === 0 ? 'var(--axis)' : 'var(--grid)' }} />
                  <text x={-6} dy="0.32em" textAnchor="end" style={AXIS_TEXT} className="tnum">{v}</text>
                </g>
              ))}
              {counts.map((c, i) => {
                const h = ih - y(c)
                return (
                  <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                    <rect x={x(i * step)} width={iw / n} height={ih} fill="transparent" />
                    {h > 0 && <path d={`M${x(i * step) + 1},${ih} v${-(h - 2)} q0,-2 2,-2 h${bw - 4} q2,0 2,2 v${h - 2} z`} style={{ fill: color }} opacity={hover == null || hover === i ? 1 : 0.55} />}
                  </g>
                )
              })}
              {x.ticks(Math.min(6, n)).map((v) => (
                <text key={v} x={x(v)} y={ih + 14} textAnchor="middle" style={AXIS_TEXT} className="tnum">{v}</text>
              ))}
            </g>
          </svg>
        )}
        {hover != null && (
          <Tip x={M.l + x(hover * step)} y={10} width={width}>
            <TipRow color={color} label={`${xLabel} ${hover * step}-${(hover + 1) * step}`} value={counts[hover]} />
          </Tip>
        )}
      </div>
    </div>
  )
}

/** Per-row tick lanes over time (e.g. technique recognized vs ground truth), two sub-lanes per row. */
export function TickLanes({ rows }: { rows: { label: string; color: string; a: number[]; b: number[] }[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<{ px: number; py: number; t: number; row: string; lane: string } | null>(null)
  const LW = 170, ROWH = 34
  const iw = Math.max(0, width - LW - 10)
  const all = rows.flatMap((r) => [...r.a, ...r.b])
  if (!all.length) return <div ref={ref} />
  const x = scaleUtc().domain([Math.min(...all), Math.max(...all) + 60_000]).range([0, iw])
  const H = rows.length * ROWH + 24
  return (
    <div ref={ref} className="relative">
      {width > 0 && (
        <svg width={width} height={H}>
          {rows.map((r, k) => (
            <g key={r.label} transform={`translate(0,${k * ROWH})`}>
              <text x={LW - 10} y={ROWH / 2} dy="0.32em" textAnchor="end" style={{ fontSize: 12, fill: 'var(--ink-2)' }}>{r.label}</text>
              <rect x={LW} y={4} width={iw} height={12} style={{ fill: 'var(--surface-2)' }} rx={2} />
              <rect x={LW} y={18} width={iw} height={12} style={{ fill: 'var(--surface-2)' }} rx={2} />
              {r.a.map((t, i) => <rect key={`a${i}`} x={LW + x(t)} y={4} width={1.6} height={12} style={{ fill: r.color }} />)}
              {r.b.map((t, i) => <rect key={`b${i}`} x={LW + x(t)} y={18} width={1.6} height={12} style={{ fill: 'var(--ink-2)' }} />)}
              <rect x={LW} y={0} width={iw} height={ROWH} fill="transparent"
                onMouseMove={(e) => {
                  const bb = e.currentTarget.getBoundingClientRect()
                  const py = e.clientY - bb.top
                  setHover({ px: e.clientX - bb.left + LW, py: k * ROWH + py, t: x.invert(e.clientX - bb.left).getTime(), row: r.label, lane: py < 17 ? 'recognized' : 'ground truth' })
                }}
                onMouseLeave={() => setHover(null)} />
            </g>
          ))}
          {x.ticks(Math.max(2, Math.floor(iw / 120))).map((t) => (
            <text key={+t} x={LW + x(t)} y={rows.length * ROWH + 14} textAnchor="middle" style={AXIS_TEXT} className="tnum">{tick(t)}</text>
          ))}
        </svg>
      )}
      {hover && (
        <Tip x={hover.px} y={hover.py} width={width}>
          <div className="font-semibold">{hover.row}</div>
          <div className="text-muted">{hover.lane} lane · {stamp(hover.t)}</div>
        </Tip>
      )}
    </div>
  )
}
