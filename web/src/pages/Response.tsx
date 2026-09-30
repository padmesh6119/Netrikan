import clsx from 'clsx'
import { ArrowCounterClockwise as Undo2, ArrowRight, Check, CheckCircle as CheckCircle2, Circle as CircleDot, Copy as ClipboardCopy, Crosshair, Eye, Lock, MagnifyingGlass as Search, ShieldCheck, ShieldWarning as ShieldAlert, Target, Wrench } from '@phosphor-icons/react'
import { useMemo, useState } from 'react'
import { DriverBars } from '../components/charts/Bars'
import { TimeChart, toBands } from '../components/charts/TimeChart'
import { Badge, Busy, Card, Empty, ErrorState, Field, Legend, Loading, Note, PageHeader, Segmented, Select, Slider, Stat, StatRow, Toolbar } from '../components/ui'
import { useApi, type DaptMeta, type Incident, type IncidentDetail, type IncidentList, type Priority, type Rung, type Z24Meta } from '../lib/api'
import { hhmm, int, num, pct, stamp, weekday } from '../lib/format'
import { useLocalState } from '../lib/store'
import { useUrlState } from '../lib/url'

type Status = 'new' | 'ack' | 'contained' | 'resolved'
type Triage = Record<string, { status: Status; done: string[] }>
const STATUS: Record<Status, { label: string; tone: 'critical' | 'warning' | 'accent' | 'good' }> = {
  new: { label: 'New', tone: 'critical' },
  ack: { label: 'Investigating', tone: 'warning' },
  contained: { label: 'Contained', tone: 'accent' },
  resolved: { label: 'Resolved', tone: 'good' },
}
const NEXT: Record<Status, { to: Status; label: string; icon: typeof Eye } | null> = {
  new: { to: 'ack', label: 'Acknowledge', icon: Eye },
  ack: { to: 'contained', label: 'Mark contained', icon: Lock },
  contained: { to: 'resolved', label: 'Resolve', icon: CheckCircle2 },
  resolved: null,
}
const PRIO_TONE: Record<Priority, string> = { P1: 'bg-critical text-white', P2: 'bg-[var(--serious)] text-white', P3: 'bg-surface-2 text-ink-2 ring-1 ring-line' }
const D3_ORDER = ['Detect', 'Isolate', 'Deceive', 'Evict', 'Harden', 'Restore', 'Model'] as const
const D3_ICON: Record<string, typeof Eye> = { Detect: Search, Isolate: Lock, Deceive: Target, Evict: ShieldAlert, Harden: Wrench, Restore: Undo2, Model: CircleDot }

function PriorityChip({ p }: { p: Priority }) {
  return <span className={clsx('inline-flex h-5 items-center rounded px-1.5 text-[11px] font-bold tracking-wide', PRIO_TONE[p])}>{p}</span>
}

export default function ResponsePage() {
  const { get, set } = useUrlState()
  const dataset = (get('ds', 'dapt') as 'dapt' | 'z24')
  const dm = useApi<DaptMeta>('/api/dapt/meta', {}, dataset === 'dapt')
  const zm = useApi<Z24Meta>('/api/z24/meta', {}, dataset === 'z24')

  const source = get('src') || (dataset === 'dapt' ? dm.data?.days[2] : zm.data?.weeks[1]?.week) || ''
  const model = get('model', dm.data?.primary ?? 'lag')
  const thr = Number(get('thr', String(dm.data?.thresholds[model as 'lag' | 'world'] ?? ''))) || undefined
  const rung = get('rung', 'sched') as Rung
  const budget = Number(get('budget', '1'))
  const params = dataset === 'dapt' ? { dataset, source, model, thr } : { dataset, source, rung, budget }
  const list = useApi<IncidentList>('/api/response/incidents', params, !!source && (dataset === 'z24' || thr != null))

  const [triage, setTriage] = useLocalState<Triage>(`netrikan-triage:${dataset}:${source}`, {})
  const statusOf = (id: string): Status => triage[id]?.status ?? 'new'
  const setStatus = (id: string, s: Status) => setTriage((t) => ({ ...t, [id]: { done: t[id]?.done ?? [], status: s } }))
  const toggleDone = (id: string, name: string) =>
    setTriage((t) => {
      const cur = t[id] ?? { status: 'new' as Status, done: [] }
      const done = cur.done.includes(name) ? cur.done.filter((d) => d !== name) : [...cur.done, name]
      return { ...t, [id]: { status: cur.status === 'new' && done.length ? 'ack' : cur.status, done } }
    })

  const [prio, setPrio] = useState<'all' | Priority>('all')
  const [statusFilter, setStatusFilter] = useState<'open' | 'all' | Status>('open')
  const [q, setQ] = useState('')
  const [showTruth, setShowTruth] = useState(false)

  const incidents = list.data?.incidents ?? []
  const filtered = useMemo(
    () =>
      incidents
        .filter((i) => prio === 'all' || i.priority === prio)
        .filter((i) => {
          const s = statusOf(i.id)
          return statusFilter === 'all' ? true : statusFilter === 'open' ? s !== 'resolved' : s === statusFilter
        })
        .filter((i) => !q || i.host.includes(q) || (i.technique ?? '').toLowerCase().includes(q.toLowerCase()) || i.tactic.toLowerCase().includes(q.toLowerCase()))
        .sort((a, b) => a.priority.localeCompare(b.priority) || b.score - a.score || a.start - b.start),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [incidents, prio, statusFilter, q, triage],
  )
  const selId = get('inc') || filtered[0]?.id || ''

  const counts = useMemo(() => {
    const c = { open: 0, P1: 0, contained: 0, resolved: 0 }
    for (const i of incidents) {
      const s = statusOf(i.id)
      if (s !== 'resolved') c.open++
      if (i.priority === 'P1' && s !== 'resolved') c.P1++
      if (s === 'contained') c.contained++
      if (s === 'resolved') c.resolved++
    }
    return c
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incidents, triage])
  const labelled = list.data?.labelled
  const confirmed = incidents.filter((i) => i.outcome === 'attack')
  const leads = confirmed.map((i) => i.lead_min ?? 0).filter((v) => v > 0).sort((a, b) => a - b)
  const horizon = dataset === 'dapt' ? (dm.data?.K ?? 5) : (zm.data?.K ?? 5)

  return (
    <>
      <Busy on={list.isFetching} />
      <PageHeader title="Incident response">
        Every forecast alert becomes an incident on one host, ranked by priority, with the MITRE ATT&amp;CK behaviour it points to, the MITRE D3FEND countermeasures that
        answer it, and containment actions filled in from the host's actual peers and ports.
      </PageHeader>

      <Toolbar>
        <Field label="Telemetry">
          <Segmented value={dataset} onChange={(v) => set({ ds: v, src: null, inc: null, thr: null, model: null })}
            items={[{ id: 'dapt', label: 'DAPT2020' }, { id: 'z24', label: 'ZeekData24' }]} />
        </Field>
        {dataset === 'dapt' && dm.data && (
          <>
            <Field label="Capture day">
              <Select value={source} onChange={(v) => set({ src: v, inc: null })} options={dm.data.days.map((d) => ({ value: d, label: `${weekday(d)} · ${dm.data!.onsets_by_day[d]} onsets` }))} />
            </Field>
            <Field label="Risk model">
              <Segmented value={model} onChange={(v) => set({ model: v, thr: null, inc: null })} items={[{ id: 'lag', label: 'LightGBM + lags' }, { id: 'world', label: 'World model' }]} />
            </Field>
            <Field label="Alert threshold">
              <div className="w-56">
                <Slider value={thr ?? 0.2} min={0.05} max={0.95} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set({ thr: v, inc: null })} />
              </div>
            </Field>
          </>
        )}
        {dataset === 'z24' && zm.data && (
          <>
            <Field label="Capture week">
              <Select value={source} onChange={(v) => set({ src: v, inc: null })} options={zm.data.weeks.map((w) => ({ value: w.week, label: `${w.week} · ${w.kind}` }))} />
            </Field>
            <Field label="Forecast model">
              <Segmented value={rung} onChange={(v) => set({ rung: v, inc: null })} items={(['sched', 'world', 'renewal'] as Rung[]).map((r) => ({ id: r, label: { sched: 'History GBDT', world: 'World model', renewal: 'Renewal baseline' }[r] }))} />
            </Field>
            <Field label="Alert budget">
              <Segmented value={budget} onChange={(v) => set({ budget: v, inc: null })} items={zm.data.budgets.map((b) => ({ id: b, label: `${b}/h`, title: `${b} false alarm(s) per attacker-hour` }))} />
            </Field>
          </>
        )}
        <label className="flex h-9 cursor-pointer items-center gap-2 self-end text-[13px] text-ink-2 lg:ml-auto">
          <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} className="size-4 accent-[var(--accent)]" />
          Show ground truth
        </label>
      </Toolbar>

      {list.error && <ErrorState error={list.error} />}
      {list.data && (
        <StatRow>
          <Stat label="Open incidents" value={int(counts.open)} hint={`${incidents.length} raised on this ${dataset === 'dapt' ? 'day' : 'week'}`} />
          <Stat label="Open P1" value={int(counts.P1)} tone={counts.P1 ? 'critical' : undefined} hint="Highest priority, not resolved" />
          <Stat label="Contained / resolved" value={`${counts.contained} / ${counts.resolved}`} hint="Your triage, saved in this browser" />
          {labelled ? (
            <>
              <Stat label="Followed by a real attack" value={pct(incidents.length ? confirmed.length / incidents.length : null)} hint={`${confirmed.length} of ${incidents.length} incidents (ground truth)`} />
              <Stat label={`Median warning (${horizon}-min horizon)`} value={leads.length ? `${leads[Math.floor(leads.length / 2)]} min` : '-'}
                hint={`Earliest alert within ${horizon} min of the attack. Missed attacks raise no incident; see the forecast pages.`} />
            </>
          ) : (
            <Stat label="Ground truth" value="-" hint="Not in this capture" />
          )}
        </StatRow>
      )}

      {list.isLoading && <Loading label="Building incidents from the forecast" />}
      {list.data && !incidents.length && (
        <Card>
          <Empty>
            No alerts at this setting, so there is nothing to respond to.
            {dataset === 'z24' && ' Benign weeks raise no attacker to forecast.'}
          </Empty>
        </Card>
      )}

      {list.data && incidents.length > 0 && (
        <div className="grid gap-5 xl:grid-cols-[400px_minmax(0,1fr)]">
          {/* ------------------------------------------------ queue */}
          <Card pad={false} className="xl:sticky xl:top-6 xl:self-start">
            <div className="space-y-3 border-b border-line p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-[15px] font-semibold">Queue</h3>
                <span className="text-[12px] text-muted tnum">{filtered.length} shown</span>
              </div>
              <div className="relative">
                <Search className="absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted" />
                <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by host, tactic or technique"
                  className="h-8 w-full rounded-lg border border-line bg-surface pr-3 pl-8 text-[13px] outline-none placeholder:text-muted focus-visible:ring-2 focus-visible:ring-accent" />
              </div>
              <div className="flex flex-wrap gap-2">
                <Segmented size="sm" value={prio} onChange={setPrio} items={[{ id: 'all', label: 'All' }, { id: 'P1', label: 'P1' }, { id: 'P2', label: 'P2' }, { id: 'P3', label: 'P3' }]} />
                <Segmented size="sm" value={statusFilter} onChange={setStatusFilter}
                  items={[{ id: 'open', label: 'Open' }, { id: 'new', label: 'New' }, { id: 'contained', label: 'Contained' }, { id: 'resolved', label: 'Resolved' }, { id: 'all', label: 'All' }]} />
              </div>
            </div>
            <ul className="max-h-[70vh] overflow-y-auto">
              {filtered.map((i) => (
                <QueueItem key={i.id} inc={i} status={statusOf(i.id)} selected={i.id === selId} showTruth={showTruth} onClick={() => set({ inc: i.id })} />
              ))}
              {!filtered.length && <li className="p-6 text-center text-[13px] text-muted">No incidents match these filters.</li>}
            </ul>
          </Card>

          {/* ------------------------------------------------ detail */}
          {selId ? (
            <IncidentView
              key={selId}
              params={{ ...params, id: selId }}
              status={statusOf(selId)}
              done={triage[selId]?.done ?? []}
              onStatus={(s) => setStatus(selId, s)}
              onToggle={(n) => toggleDone(selId, n)}
              showTruth={showTruth}
              dataset={dataset}
            />
          ) : (
            <Card>
              <Empty>Select an incident from the queue.</Empty>
            </Card>
          )}
        </div>
      )}
      {list.data && <Note>{list.data.note} Triage status and checklists are saved in this browser only.</Note>}
    </>
  )
}

function QueueItem({ inc, status, selected, showTruth, onClick }: { inc: Incident; status: Status; selected: boolean; showTruth: boolean; onClick: () => void }) {
  return (
    <li>
      <button onClick={onClick}
        className={clsx('flex w-full gap-3 border-b border-line px-4 py-3 text-left transition-colors', selected ? 'bg-accent-wash' : 'hover:bg-surface-hover', status === 'resolved' && 'opacity-60')}>
        <div className="pt-0.5"><PriorityChip p={inc.priority} /></div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate font-mono text-[13px] font-medium">{inc.host}</span>
            <Badge tone={STATUS[status].tone}>{STATUS[status].label}</Badge>
          </div>
          <div className="mt-0.5 truncate text-[12.5px] text-ink-2">
            {inc.technique ? `${inc.technique} · ` : ''}
            {inc.tactic}
          </div>
          <div className="mt-1 flex items-center gap-3 text-[11.5px] text-muted tnum">
            <span>{stamp(inc.start)}-{hhmm(inc.end)}</span>
            <span>peak {pct(inc.peak)}</span>
            <span>{inc.minutes} min</span>
            {showTruth && inc.outcome && (
              <span className={clsx('ml-auto inline-flex items-center gap-1', inc.outcome === 'attack' ? 'text-critical' : 'text-muted')}>
                <span className={clsx('size-1.5 rounded-full', inc.outcome === 'attack' ? 'bg-critical' : 'bg-[var(--neutral-series)]')} />
                {inc.outcome === 'attack' ? 'attack' : 'no attack'}
              </span>
            )}
          </div>
        </div>
      </button>
    </li>
  )
}

function IncidentView({ params, status, done, onStatus, onToggle, showTruth, dataset }: {
  params: Record<string, string | number | undefined>
  status: Status
  done: string[]
  onStatus: (s: Status) => void
  onToggle: (name: string) => void
  showTruth: boolean
  dataset: 'dapt' | 'z24'
}) {
  const d = useApi<IncidentDetail>('/api/response/incident', params)
  const [copied, setCopied] = useState<string | null>(null)
  if (d.error) return <ErrorState error={d.error} />
  if (!d.data) return <Card><Loading label="Loading incident" /></Card>
  const { incident: inc, playbook, targets: tg, drivers, rules, series, confidence } = d.data
  const next = NEXT[status]
  const allItems = playbook.flatMap((g) => g.items)
  const progress = allItems.filter((c) => done.includes(c.d3fend + c.action)).length

  const copy = async (what: 'rules' | 'report') => {
    const text = what === 'rules' ? rules.join('\n') : report(d.data!)
    try {
      await navigator.clipboard.writeText(text)
      setCopied(what)
      setTimeout(() => setCopied(null), 1500)
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <div className="min-w-0 space-y-5">
      {/* header */}
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <PriorityChip p={inc.priority} />
              <Badge tone={STATUS[status].tone}>{STATUS[status].label}</Badge>
              {showTruth && inc.outcome && (
                <Badge tone={inc.outcome === 'attack' ? 'critical' : 'neutral'}>
                  Ground truth: {inc.outcome === 'attack' ? `attack followed${inc.true_tactic ? ` (${inc.true_tactic})` : ''}, ${inc.lead_min ? `warned ${inc.lead_min} min ahead` : 'no alert in the 5 min before it'}` : 'no attack followed'}
                </Badge>
              )}
            </div>
            <h2 className="mt-2 font-mono text-[20px] font-semibold tracking-tight">{inc.host}</h2>
            <p className="mt-0.5 text-[13px] text-ink-2 tnum">
              Alerting {stamp(inc.start)}-{hhmm(inc.end)} · {inc.minutes} alert-window minutes · peak risk {pct(inc.peak)} ({num(inc.ratio, 1)}× threshold)
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button onClick={() => copy('report')} className="btn btn-quiet">
              {copied === 'report' ? <Check className="size-3.5" /> : <ClipboardCopy className="size-3.5" />} {copied === 'report' ? 'Copied' : 'Copy report'}
            </button>
            {status !== 'new' && (
              <button onClick={() => onStatus('new')} className="btn btn-quiet">
                <Undo2 className="size-3.5" /> Reopen
              </button>
            )}
            {next && (
              <button onClick={() => onStatus(next.to)} className="btn btn-primary">
                <next.icon className="size-3.5" /> {next.label}
              </button>
            )}
          </div>
        </div>
      </Card>

      {/* ATT&CK -> D3FEND */}
      <Card title="Threat → countermeasures" subtitle="MITRE ATT&CK behaviour the forecast points to, and the MITRE D3FEND techniques that answer it"
        actions={<span className="text-[12px] text-muted tnum">{progress}/{allItems.length} actions done</span>}>
        <div className="grid gap-4 lg:grid-cols-[220px_auto_minmax(0,1fr)] lg:items-start">
          <div className="rounded-lg border border-line bg-surface-2 p-4">
            <div className="text-[12px] font-medium text-muted">MITRE ATT&amp;CK</div>
            <div className="mt-1.5 text-[15px] font-semibold">{inc.tactic}</div>
            {inc.technique && (
              <div className="mt-0.5 text-[13px] text-ink-2">
                <span className="font-mono">{inc.technique}</span> {d.data.attack_ref.technique_name}
              </div>
            )}
            {inc.tactic_p != null && <div className="mt-1 text-[12px] text-muted">Stage model: {pct(inc.tactic_p)} for this stage</div>}
            {inc.recent && inc.recent.length > 0 && (
              <div className="mt-3">
                <div className="text-[11px] text-muted">Seen from this host in the last hour</div>
                <div className="mt-1 flex flex-wrap gap-1">{inc.recent.map((t) => <Badge key={t}>{t}</Badge>)}</div>
              </div>
            )}
            <div className={clsx('mt-3 flex items-start gap-1.5 text-[12px]', confidence.level === 'low' ? 'text-ink-2' : 'text-good-ink')}>
              {confidence.level === 'low' ? <ShieldAlert className="mt-0.5 size-3.5 shrink-0 text-warning" /> : <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />}
              <span>{confidence.level === 'low' ? 'Low confidence in the stage' : 'High confidence in the technique'}</span>
            </div>
          </div>
          <ArrowRight className="mx-auto mt-10 hidden size-5 text-muted lg:block" />
          <div className="space-y-5">
            {playbook.map((g) => (
              <div key={g.group}>
                <div className="mb-2 flex items-center gap-2">
                  <h4 className="text-[13px] font-semibold">{g.group}</h4>
                  {g.confidence === 'low' && <Badge tone="warning">hypothesis</Badge>}
                </div>
                <div className="grid gap-2.5 md:grid-cols-2">
                  {[...g.items].sort((a, b) => D3_ORDER.indexOf(a.tactic) - D3_ORDER.indexOf(b.tactic)).map((c) => {
                    const key = c.d3fend + c.action
                    const isDone = done.includes(key)
                    const Icon = D3_ICON[c.tactic] ?? CircleDot
                    return (
                      <button key={key} onClick={() => onToggle(key)}
                        className={clsx('group flex gap-3 rounded-lg border p-3 text-left transition-colors', isDone ? 'border-good/40 bg-good/6' : 'border-line hover:border-line-strong hover:bg-surface-hover')}>
                        <span className={clsx('mt-0.5 flex size-4 shrink-0 items-center justify-center rounded border', isDone ? 'border-good bg-good text-white' : 'border-line-strong')}>
                          {isDone && <Check className="size-3" />}
                        </span>
                        <span className="min-w-0">
                          <span className="flex flex-wrap items-center gap-1.5">
                            <span className="inline-flex items-center gap-1 rounded-[5px] bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium text-ink-2">
                              <Icon className="size-3" /> {c.tactic}
                            </span>
                            <span className="text-[13px] font-semibold">{c.d3fend}</span>
                          </span>
                          <span className={clsx('mt-1 block text-[12.5px] leading-snug', isDone ? 'text-ink-2 line-through decoration-muted' : 'text-ink')}>{c.action}</span>
                          <span className="mt-1 block text-[11.5px] leading-snug text-muted">{c.why}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
        {confidence.level === 'low' && (
          <div className="mt-4">
            <Note tone="caveat" title="Why the stage playbook is marked as a hypothesis">{confidence.text} Start with the stage-agnostic first response.</Note>
          </div>
        )}
        {confidence.level === 'high' && <p className="mt-4 text-[12px] text-muted">{confidence.text}</p>}
      </Card>

      {/* context */}
      <Card title="Risk around the incident" subtitle={dataset === 'dapt' ? 'P(attack in the next 5 min) for this host, ±30 min' : `P(next ${inc.technique} burst within 5 min) for this attacker, ±30 min`}
        actions={<Legend items={[{ label: 'Risk', color: 'var(--s1)', shape: 'line' }, { label: 'Incident window', color: 'var(--ink-2)' }, ...(showTruth ? [{ label: 'True attack minutes', color: 'var(--critical)' }] : [])]} />}>
        <TimeChart
          height={220}
          yMax={1}
          yFormat={(v) => pct(v)}
          yLabel="risk"
          series={[{ id: 'p', label: 'Risk', color: 'var(--s1)', t: series.t, v: series.p }]}
          bands={[
            { start: inc.start, end: inc.end, color: 'var(--ink-2)', label: 'Incident window' },
            ...(showTruth ? toBands(series.t, series.attack, () => 'var(--critical)', () => 'True attack') : []),
          ]}
        />
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Why the model alerted" subtitle="TreeSHAP contributions at the peak minute">
          <DriverBars items={drivers} />
        </Card>
        <Card title="Who it talked to" subtitle={`${int(tg.flows)} flows from 5 min before the alert to 5 min after · ${int(tg.bytes_out)} B out / ${int(tg.bytes_in)} B in`}>
          <div className="grid gap-4 sm:grid-cols-2">
            <MiniList title="Outbound peers" rows={tg.out_peers} mono />
            <MiniList title="Outbound dst ports" rows={tg.out_ports} mono />
            <MiniList title="Inbound peers" rows={tg.in_peers} mono />
            <MiniList title="Inbound dst ports" rows={tg.in_ports} mono />
          </div>
        </Card>
      </div>

      <Card title="Containment rules" subtitle="Example iptables rules generated from this incident. Templates: review and adapt before applying."
        actions={
          <button onClick={() => copy('rules')} className="btn btn-quiet h-8">
            {copied === 'rules' ? <Check className="size-3.5" /> : <ClipboardCopy className="size-3.5" />} {copied === 'rules' ? 'Copied' : 'Copy'}
          </button>
        }>
        <pre className="overflow-x-auto rounded-lg bg-surface-2 p-4 font-mono text-[12.5px] leading-relaxed text-ink-2">
          {rules.map((r) => (
            <div key={r} className={r.startsWith('#') ? 'text-muted' : 'text-ink'}>{r}</div>
          ))}
        </pre>
        <p className="mt-2 flex items-center gap-1.5 text-[12px] text-muted"><Crosshair className="size-3.5" /> Netrikan never applies rules itself; it runs offline and only recommends.</p>
      </Card>
    </div>
  )
}

function MiniList({ title, rows, mono }: { title: string; rows: { value: string; flows: number }[]; mono?: boolean }) {
  const max = Math.max(1, ...rows.map((r) => r.flows))
  return (
    <div>
      <div className="mb-1.5 text-[11.5px] font-semibold text-muted">{title}</div>
      {!rows.length && <div className="text-[12.5px] text-muted">None</div>}
      <div className="space-y-1">
        {rows.map((r) => (
          <div key={r.value} className="relative flex items-center justify-between overflow-hidden rounded px-2 py-1 text-[12.5px]">
            <div className="absolute inset-y-0 left-0 bg-accent-wash" style={{ width: `${(r.flows / max) * 100}%` }} />
            <span className={clsx('relative truncate', mono && 'font-mono')}>{r.value}</span>
            <span className="relative text-muted tnum">{int(r.flows)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function report(d: IncidentDetail) {
  const i = d.incident
  const lines = [
    `# ${i.priority} incident · ${i.host}`,
    `Window: ${stamp(i.start)}-${hhmm(i.end)} (${i.minutes} min), peak risk ${pct(i.peak)} (${num(i.ratio, 1)}× threshold)`,
    `ATT&CK: ${i.tactic}${i.technique ? ` · ${i.technique} ${d.attack_ref.technique_name}` : ''} (${d.confidence.level} confidence)`,
    '',
    '## Why',
    ...d.drivers.map((x) => `- ${x.feature}: ${x.value >= 0 ? '+' : ''}${x.value.toFixed(2)} log-odds`),
    '',
    '## D3FEND countermeasures',
    ...d.playbook.flatMap((g) => [`### ${g.group}${g.confidence === 'low' ? ' (hypothesis)' : ''}`, ...g.items.map((c) => `- [${c.tactic}] ${c.d3fend}: ${c.action}`)]),
    '',
    '## Containment (template, review before applying)',
    '```',
    ...d.rules,
    '```',
  ]
  return lines.join('\n')
}

