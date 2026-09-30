// Timestamps are the captures' wall-clock time, sent as epoch ms; always format in UTC so the clock time is preserved.
const two = (n: number) => String(n).padStart(2, '0')
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export const hhmm = (t: number) => {
  const d = new Date(t)
  return `${two(d.getUTCHours())}:${two(d.getUTCMinutes())}`
}
export const dayLabel = (t: number) => {
  const d = new Date(t)
  return `${d.getUTCDate()} ${MON[d.getUTCMonth()]}`
}
export const stamp = (t: number) => `${dayLabel(t)} ${hhmm(t)}`
export const weekday = (iso: string) => new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' })

export const pct = (v: number | null | undefined, digits = 0) => (v == null || Number.isNaN(v) ? '-' : `${(v * 100).toFixed(digits)}%`)
export const num = (v: number | null | undefined, digits = 2) => (v == null || Number.isNaN(v) ? '-' : v.toFixed(digits))
export const int = (v: number | null | undefined) => (v == null ? '-' : Math.round(v).toLocaleString('en-US'))
export const signed = (v: number, digits = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`

/** Axis ticks: 24-hour clock, with the date at midnight. */
export const tick = (d: Date | number) => {
  const t = +d
  const x = new Date(t)
  return x.getUTCHours() === 0 && x.getUTCMinutes() === 0 ? dayLabel(t) : hhmm(t)
}
