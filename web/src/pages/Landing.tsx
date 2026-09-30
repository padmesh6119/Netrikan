import { ArrowRight, Brain, ChartLineUp, Crosshair, Graph, ListMagnifyingGlass, Waveform } from '@phosphor-icons/react'
import { animate, motion, useMotionValue, useReducedMotion } from 'motion/react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { EYE_ASPECT, EYE_PROPS, REPLAY_INTRO } from '../components/brand'
import { TimeChart, toBands } from '../components/charts/TimeChart'
import EvilEye from '../components/EvilEye'
import { Legend } from '../components/ui'
import { useApi, type DaptMeta, type DaptScores, type IncidentDetail, type IncidentList, type ZeroShot } from '../lib/api'
import { STAGE_COLOR } from '../lib/colors'
import { num, pct } from '../lib/format'
import { useResolvedTheme } from '../lib/theme'

const EASE = [0.16, 1, 0.3, 1] as const
const HERO_HOST = '192.168.3.29'
const HERO_DAY = '2019-07-16'

/** Fade-up on first entry into view; static under reduced motion. */
function Reveal({ children, delay = 0, className }: { children: ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion()
  return (
    <motion.div className={className} initial={reduce ? false : { opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.6, delay, ease: EASE }}>
      {children}
    </motion.div>
  )
}

const wrap = 'mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-10'

// ---------------------------------------------------------------- intro: the eye opens large, then springs into the header
// Plays once per page load ("boot"). The first scroll, key, swipe or click docks the eye with a spring and collapses the
// intro screen so the hero rises into view. After that the eye stays docked, also when scrolling back up.
let introDone = false

interface Geo { W: number; H: number; sx: number; sy: number; ex: number; ey: number; es: number }

function measure(): Geo {
  const vw = window.innerWidth, vh = window.innerHeight
  const W = Math.min(vw * 0.92, 920)
  const H = W / EYE_ASPECT
  const slot = document.getElementById('eye-slot')?.getBoundingClientRect()
  return {
    W, H,
    sx: (vw - W) / 2, sy: vh * 0.42 - H / 2, // intro: centred, slightly above the middle
    ex: slot?.left ?? (vw - 144) / 2, ey: slot?.top ?? 0, es: (slot?.width ?? 144) / W, // docked: exactly on the header slot
  }
}

const SPRING = { type: 'spring', stiffness: 140, damping: 22, mass: 1 } as const

function Intro() {
  const theme = useResolvedTheme()
  const reduce = useReducedMotion()
  const [docked, setDocked] = useState(introDone)
  const [geo, setGeo] = useState<Geo | null>(null)
  const x = useMotionValue(0), y = useMotionValue(0), scale = useMotionValue(1)
  const mounted = useRef(false)

  // Place the eye: at the intro position, or on the header slot once docked (resize keeps it there).
  useEffect(() => {
    const place = (animateIt: boolean) => {
      const g = measure()
      setGeo(g)
      const to = docked ? { x: g.ex, y: g.ey, scale: g.es } : { x: g.sx, y: g.sy, scale: 1 }
      if (animateIt && !reduce) {
        animate(x, to.x, SPRING)
        animate(y, to.y, SPRING)
        animate(scale, to.scale, SPRING)
      } else {
        x.set(to.x)
        y.set(to.y)
        scale.set(to.scale)
      }
    }
    place(mounted.current)
    mounted.current = true
    const onResize = () => place(false)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [docked, reduce, x, y, scale])

  // Clicking the header eye on this page replays the intro: back to the top, the eye springs down to the centre.
  useEffect(() => {
    const replay = () => {
      introDone = false
      window.scrollTo({ top: 0, behavior: 'instant' })
      setDocked(false)
    }
    window.addEventListener(REPLAY_INTRO, replay)
    return () => window.removeEventListener(REPLAY_INTRO, replay)
  }, [])

  // While the intro shows, the page doesn't scroll: the first intent to move on docks the eye instead.
  useEffect(() => {
    if (docked) return
    const root = document.documentElement
    window.scrollTo(0, 0)
    root.style.overflow = 'hidden'
    const dock = () => {
      introDone = true
      setDocked(true)
    }
    const onWheel = (e: WheelEvent) => e.deltaY > 4 && dock()
    let startY = 0
    const onTouchStart = (e: TouchEvent) => { startY = e.touches[0].clientY }
    const onTouchMove = (e: TouchEvent) => startY - e.touches[0].clientY > 12 && dock()
    const onKey = (e: KeyboardEvent) => ['ArrowDown', 'PageDown', ' ', 'End', 'Enter'].includes(e.key) && dock()
    // Any scroll that still gets through (scrollbar drag, programmatic scroll, anchor jump) also docks.
    const onScroll = () => window.scrollY > 2 && dock()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('wheel', onWheel, { passive: true })
    window.addEventListener('touchstart', onTouchStart, { passive: true })
    window.addEventListener('touchmove', onTouchMove, { passive: true })
    window.addEventListener('keydown', onKey)
    return () => {
      root.style.overflow = ''
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('wheel', onWheel)
      window.removeEventListener('touchstart', onTouchStart)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('keydown', onKey)
    }
  }, [docked])

  return (
    <>
      {geo && (
        <motion.div aria-hidden className="pointer-events-none fixed top-0 left-0 z-40" style={{ width: geo.W, height: geo.H, x, y, scale, transformOrigin: '0 0', ...(theme === 'light' && { filter: 'grayscale(1) invert(1) brightness(0.4)' }) }}
          initial={{ opacity: introDone ? 1 : 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.2, ease: EASE }}>
          <EvilEye className="h-full w-full" {...EYE_PROPS} lightMode={false} />
        </motion.div>
      )}
      {/* The intro screen collapses as the eye docks, so the hero rises into place with no empty gap. */}
      <motion.section aria-label="Netrikan" className="relative overflow-hidden" initial={false}
        animate={{ height: docked ? 0 : '100dvh' }} transition={reduce ? { duration: 0 } : { duration: 0.75, ease: EASE }}
        onClick={() => { introDone = true; setDocked(true) }}>
        <motion.div animate={{ opacity: docked ? 0 : 1, y: docked ? -24 : 0 }} transition={{ duration: 0.35 }}
          className="absolute inset-x-0 top-[calc(42dvh+min(92vw,920px)/4.8+20px)] px-4 text-center">
          <p className="text-[34px] font-semibold tracking-[-0.03em] sm:text-[44px]">Netrikan</p>
          <p className="mt-2 text-[15px] text-ink-2">Network attack forecasting, offline.</p>
        </motion.div>
      </motion.section>
    </>
  )
}

// ---------------------------------------------------------------- hero: a real forecast from a held-out day
function LiveForecast() {
  const meta = useApi<DaptMeta>('/demo-meta.json')
  const sc = useApi<DaptScores>('/demo-scores.json')
  const d = useMemo(() => {
    if (!sc.data || !meta.data) return null
    const hi = sc.data.hosts.indexOf(HERO_HOST)
    const idx = sc.data.rows.host.flatMap((h, k) => (h === hi ? [k] : []))
    const t = idx.map((k) => sc.data!.rows.t[k])
    const thr = meta.data.thresholds.lag
    return {
      t, thr,
      p: idx.map((k) => sc.data!.rows.lag[k]),
      bands: toBands(t, idx.map((k) => sc.data!.rows.stage[k]), (c) => STAGE_COLOR[meta.data!.stages[c as number]], (c) => meta.data!.stages[c as number]),
      alerts: idx.filter((k) => sc.data!.rows.lag[k] >= thr).map((k) => ({ t: sc.data!.rows.t[k], v: sc.data!.rows.lag[k], color: 'var(--critical)' })),
    }
  }, [sc.data, meta.data])
  return (
    <figure className="rounded-2xl border border-line bg-surface p-4 shadow-[0_24px_60px_-30px_rgba(20,40,80,0.35)] sm:p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="text-[13px] font-medium">P(attack in the next 5 min)</div>
        <Legend items={[{ label: 'Forecast', color: 'var(--s1)', shape: 'line' }, { label: 'Alert', color: 'var(--critical)', shape: 'dot' }, { label: 'Reconnaissance', color: STAGE_COLOR.Reconnaissance }]} />
      </div>
      {d ? (
        <TimeChart height={280} yFormat={(v) => pct(v)} series={[{ id: 'p', label: 'Forecast', color: 'var(--s1)', t: d.t, v: d.p }]} bands={d.bands} points={d.alerts}
          threshold={{ v: d.thr, label: 'alert threshold' }} />
      ) : (
        <div className="skeleton h-[280px] w-full" />
      )}
      <figcaption className="mt-3 text-[12.5px] leading-relaxed text-muted">
        Host {HERO_HOST}, 16 Jul 2019, scored by a model that never saw that day. Shaded minutes are the real reconnaissance.
      </figcaption>
    </figure>
  )
}

function Hero() {
  const reduce = useReducedMotion()
  const item = (i: number) => ({ initial: reduce ? false : { opacity: 0, y: 16 }, whileInView: { opacity: 1, y: 0 }, viewport: { once: true, amount: 0.3 }, transition: { duration: 0.7, delay: 0.08 * i, ease: EASE } })
  return (
    <section className={`${wrap} grid items-center gap-10 pt-14 pb-20 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-14 lg:pt-20 lg:pb-28`}>
      <div>
        <motion.h1 {...item(0)} className="text-[40px] leading-[1.05] font-semibold tracking-[-0.035em] sm:text-5xl lg:text-[56px]">
          See the attack before it starts.
        </motion.h1>
        <motion.p {...item(1)} className="mt-5 max-w-[46ch] text-[17px] leading-relaxed text-ink-2">
          Netrikan forecasts which host is about to be attacked, explains why, and hands analysts a D3FEND response plan.
        </motion.p>
        <motion.div {...item(2)} className="mt-8 flex flex-wrap items-center gap-5">
          <Link to={`/dashboard/dapt?src=${HERO_DAY}&tab=investigate&host=${HERO_HOST}`} className="btn btn-primary h-11 px-5 text-[14px]">
            See a live forecast <ArrowRight className="size-4" />
          </Link>
          <Link to="/dashboard/zero-shot" className="text-[14px] font-medium text-ink-2 underline decoration-line-strong underline-offset-4 transition-colors hover:text-ink hover:decoration-ink">
            Read the evidence
          </Link>
        </motion.div>
      </div>
      <motion.div {...item(3)}>
        <LiveForecast />
      </motion.div>
    </section>
  )
}

// ---------------------------------------------------------------- how it works: sticky heading, stage list
const STAGES = [
  { icon: Waveform, title: 'Ingest flows', body: 'CICFlowMeter, Zeek and Argus records. No packet payloads, no cloud calls.' },
  { icon: Graph, title: 'Model each host', body: 'Every monitored host becomes a state vector per minute: peers, ports, bytes, flags, service mix.' },
  { icon: Brain, title: 'Learn the dynamics', body: 'A GRU world model learns how that state evolves and rolls it five minutes forward.' },
  { icon: ChartLineUp, title: 'Forecast', body: 'A gradient-boosted head turns state and simulated future into P(attack soon), stage and ATT&CK technique.' },
  { icon: ListMagnifyingGlass, title: 'Explain', body: 'Exact TreeSHAP attributions for every minute, plus what the world model simulated against what happened.' },
  { icon: Crosshair, title: 'Respond', body: 'Alerts merge into incidents with D3FEND countermeasures filled in from the host’s real peers and ports.' },
]

function HowItWorks() {
  return (
    <section className="border-t border-line bg-surface">
      <div className={`${wrap} grid gap-12 py-20 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)] lg:py-28`}>
        <div className="lg:sticky lg:top-28 lg:self-start">
          <h2 className="text-[32px] leading-[1.1] font-semibold tracking-[-0.03em] lg:text-[40px]">One code path, from flow to action.</h2>
          <p className="mt-4 max-w-[42ch] text-[15px] leading-relaxed text-ink-2">The same code trains the models, runs the benchmark and serves the dashboard, so what you see is what was measured.</p>
        </div>
        <ol className="grid gap-x-10 gap-y-10 sm:grid-cols-2">
          {STAGES.map((s, i) => (
            <Reveal key={s.title} delay={0.05 * i}>
              <li className="list-none">
                <s.icon className="size-6 text-accent" />
                <h3 className="mt-3 text-[16px] font-semibold tracking-[-0.01em]">{s.title}</h3>
                <p className="mt-1.5 text-[14px] leading-relaxed text-ink-2">{s.body}</p>
              </li>
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- evidence bento: 4 cells, all live numbers
interface DM { pooled: Record<string, Record<string, number>>; stage: { accuracy: number; majority_baseline_accuracy: number }; dataset: Record<string, number> }
interface ZM { forecast: { macro: Record<string, Record<string, number>> } }

function Evidence() {
  const dm = useApi<DM>('/demo-dapt-metrics.json')
  const zm = useApi<ZM>('/demo-z24-metrics.json')
  const zs = useApi<ZeroShot>('/demo-zeroshot.json')
  const lag = dm.data?.pooled.lag, prev = dm.data?.pooled.lr.prevalence
  const f = zm.data?.forecast.macro
  const mean = (k: 'iforest' | 'gbdt') => (zs.data ? zs.data.rows.reduce((a, r) => a + (r.auc[k] ?? 0), 0) / zs.data.rows.length : undefined)
  const iso = mean('iforest'), sup = mean('gbdt')
  const shuf = f ? (f.sched.pr_auc - f.sched_shuf.pr_auc) / f.sched.pr_auc : undefined
  const Skel = ({ w = 'w-24' }: { w?: string }) => <span className={`skeleton inline-block h-10 align-middle ${w}`} />
  return (
    <section className={`${wrap} py-20 lg:py-28`}>
      <h2 className="max-w-[22ch] text-[32px] leading-[1.1] font-semibold tracking-[-0.03em] lg:text-[40px]">What the evidence supports.</h2>
      <p className="mt-4 max-w-[60ch] text-[15px] leading-relaxed text-ink-2">Every figure below is read live from held-out evaluation: days, weeks or whole labs the models never trained on.</p>
      <div className="mt-12 grid gap-4 md:grid-cols-3 md:grid-rows-[auto_auto]">
        <Reveal className="md:col-span-2 md:row-span-2">
          <article className="flex h-full flex-col justify-between rounded-2xl border border-line bg-surface p-7 lg:p-9">
            <div>
              <div className="text-[64px] leading-none font-medium tracking-[-0.04em] tnum lg:text-[88px]">{lag && prev ? `${num(lag.pr_auc / prev, 1)}x` : <Skel w="w-40" />}</div>
              <p className="mt-4 max-w-[44ch] text-[16px] leading-relaxed text-ink-2">better than chance at ranking which DAPT2020 host is about to be attacked, on days the model never saw.</p>
            </div>
            {lag && prev && (
              <dl className="mt-10 space-y-3 text-[13px]">
                {[{ k: 'Netrikan forecast', v: lag.pr_auc, c: 'var(--accent)' }, { k: 'Random scorer (base rate)', v: prev, c: 'var(--neutral-series)' }].map((r) => (
                  <div key={r.k} className="grid grid-cols-[12rem_1fr_3rem] items-center gap-3">
                    <dt className="text-ink-2">{r.k}</dt>
                    <dd className="h-2 rounded-r-[4px]" style={{ width: `${(r.v / lag.pr_auc) * 100}%`, background: r.c }} />
                    <dd className="text-right font-mono text-ink tnum">{num(r.v)}</dd>
                  </div>
                ))}
                <p className="pt-1 text-[12px] text-muted">PR-AUC, leave-one-day-out, pooled over five capture days.</p>
              </dl>
            )}
          </article>
        </Reveal>
        <Reveal delay={0.08}>
          <article className="h-full rounded-2xl border border-line bg-accent-wash p-7">
            <div className="text-[44px] leading-none font-medium tracking-[-0.03em] tnum">{f ? `${num(f.sched.pr_auc / f.sched.prevalence, 1)}x` : <Skel />}</div>
            <p className="mt-3 text-[14px] leading-relaxed text-ink-2">
              lift at forecasting each ZeekData24 attacker's next burst. Shuffle its history order and skill drops {shuf != null ? pct(shuf) : '...'}, so the model uses time, not just state.
            </p>
          </article>
        </Reveal>
        <Reveal delay={0.16}>
          <article className="h-full rounded-2xl border border-line bg-surface p-7">
            <div className="text-[14px] font-medium">Attacks from labs it never saw</div>
            {iso != null && sup != null ? (
              <div className="relative mt-8 mb-6 h-10">
                <div className="absolute inset-x-0 top-1/2 h-px bg-line-strong" />
                <div className="absolute top-1/2 h-4 w-px -translate-y-1/2 bg-ink-2" style={{ left: '50%' }} />
                <span className="absolute top-full mt-1 -translate-x-1/2 text-[11px] text-muted" style={{ left: '50%' }}>chance</span>
                {[{ v: sup, c: 'var(--neutral-series)', l: 'Supervised' }, { v: iso, c: 'var(--accent)', l: 'Anomaly' }].map((r) => (
                  <span key={r.l} className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2" style={{ left: `${r.v * 100}%` }}>
                    <span className="block size-3.5 rounded-full ring-2 ring-surface" style={{ background: r.c }} />
                    <span className="absolute bottom-full left-1/2 mb-1.5 -translate-x-1/2 text-[11.5px] whitespace-nowrap text-ink-2">{r.l} {num(r.v)}</span>
                  </span>
                ))}
              </div>
            ) : <div className="skeleton mt-6 h-10" />}
            <p className="text-[13.5px] leading-relaxed text-ink-2">Mean ROC-AUC on 13 unseen attack families. Learning other labs' attacks barely beats chance; benign-only anomaly scoring transfers.</p>
          </article>
        </Reveal>
        <Reveal className="md:col-span-3" delay={0.1}>
          <article className="grid gap-4 rounded-2xl border border-warning/35 bg-warning/8 p-7 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] md:items-center">
            <div className="text-[18px] font-semibold tracking-[-0.01em]">Where it stops.</div>
            <p className="text-[14px] leading-relaxed text-ink-2">
              Stage forecasting is at chance across held-out days ({dm.data ? `${pct(dm.data.stage.accuracy)} vs ${pct(dm.data.stage.majority_baseline_accuracy)} for the majority class` : '...'}),
              and the world model matches gradient-boosted trees rather than beating them. The dashboard says so wherever it matters.
            </p>
          </article>
        </Reveal>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- response: a real incident, live
function LiveIncident() {
  const list = useApi<IncidentList>('/demo-incidents.json')
  const top = list.data?.incidents.reduce((a, b) => (b.score > a.score ? b : a), list.data.incidents[0])
  const d = useApi<IncidentDetail>('/demo-incident-detail.json', {}, !!top)
  if (!d.data) return <div className="skeleton h-[360px] rounded-2xl" />
  const i = d.data.incident
  const items = d.data.playbook.flatMap((g) => g.items.map((c) => ({ ...c, hyp: g.confidence === 'low' })))
  return (
    <div className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="flex items-center gap-2">
        <span className="rounded-[5px] bg-critical px-1.5 py-0.5 text-[11px] font-bold text-white">{i.priority}</span>
        <span className="font-mono text-[15px] font-medium">{i.host}</span>
        <span className="ml-auto text-[12px] text-muted">peak risk {pct(i.peak)}</span>
      </div>
      <div className="mt-1 text-[13px] text-ink-2">{i.tactic} (stage is a hypothesis on this corpus)</div>
      <ul className="mt-5 space-y-3">
        {items.slice(0, 4).map((c) => (
          <li key={c.d3fend + c.action} className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-3">
            <span className="h-fit rounded-[5px] bg-surface-2 px-1.5 py-0.5 text-center text-[11px] font-medium text-ink-2">{c.tactic}</span>
            <span className="min-w-0">
              <span className="block text-[13.5px] font-medium">{c.d3fend}</span>
              <span className="block truncate text-[12.5px] text-muted">{c.action}</span>
            </span>
          </li>
        ))}
      </ul>
      <pre className="mt-5 overflow-x-auto rounded-lg bg-surface-2 p-3 font-mono text-[11.5px] leading-relaxed text-ink-2">{d.data.rules.filter((r) => !r.startsWith('#')).slice(0, 3).join('\n')}</pre>
    </div>
  )
}

function Response() {
  return (
    <section className="border-t border-line">
      <div className={`${wrap} grid items-center gap-12 py-20 lg:grid-cols-[minmax(0,6fr)_minmax(0,5fr)] lg:py-28`}>
        <Reveal className="lg:order-2">
          <LiveIncident />
        </Reveal>
        <div className="lg:order-1">
          <h2 className="text-[32px] leading-[1.1] font-semibold tracking-[-0.03em] lg:text-[40px]">From alert to action in one queue.</h2>
          <p className="mt-4 max-w-[52ch] text-[15px] leading-relaxed text-ink-2">
            Alerts on the same host merge into an incident. Each one maps its MITRE ATT&amp;CK behaviour to MITRE D3FEND countermeasures, names the peers and ports to act on, and drafts
            containment rules for an analyst to review. Nothing is applied automatically.
          </p>
          <Link to="/dashboard/response" className="mt-7 inline-flex items-center gap-1.5 text-[14px] font-medium text-accent-ink underline-offset-4 hover:underline">
            Open incident response <ArrowRight className="size-4" />
          </Link>
          <p className="mt-10 max-w-[52ch] text-[12.5px] leading-relaxed text-muted">Shown: the highest-priority incident on DAPT2020, Wed 17 Jul, computed live.</p>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="border-t border-line">
      <div className={`${wrap} flex flex-wrap items-center justify-between gap-4 py-8 text-[12.5px] text-muted`}>
        <span>Netrikan runs offline on flow telemetry. Built for Smart India Hackathon problem SIH26153.</span>
        <span className="flex gap-5">
          <Link to="/dashboard" className="hover:text-ink">Dashboard</Link>
          <a href="/api/docs" target="_blank" rel="noreferrer" className="hover:text-ink">API docs</a>
        </span>
      </div>
    </footer>
  )
}

export default function LandingPage() {
  return (
    <>
      <Intro />
      <Hero />
      <HowItWorks />
      <Evidence />
      <Response />
      <Footer />
    </>
  )
}
