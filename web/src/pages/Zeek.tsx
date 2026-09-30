import clsx from 'clsx'
import { ShieldCheck } from '@phosphor-icons/react'
import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { DriverBars, ProbBars } from '../components/charts/Bars'
import { Histogram, TickLanes } from '../components/charts/Small'
import { TimeChart } from '../components/charts/TimeChart'
import { ScoreBar, Table } from '../components/Table'
import { Badge, Busy, Card, Empty, ErrorState, Field, Findings, Legend, Loading, Note, PageHeader, Segmented, Select, Slider, Stat, StatRow, Tabs, Toolbar } from '../components/ui'
import { useApi, type Rung, type Z24Forecast, type Z24Meta, type Z24RecogExplain, type Z24RecogHost, type Z24Structure, type Z24Week } from '../lib/api'
import { seq, TECH_COLOR } from '../lib/colors'
import { int, num, pct, stamp, weekday } from '../lib/format'
import { useUrlState } from '../lib/url'

type Tab = 'recognize' | 'forecast' | 'structure' | 'evidence'
const RUNG_SHORT: Record<Rung, string> = { sched: 'History GBDT', world: 'World model', renewal: 'Renewal baseline' }

export default function ZeekPage() {
  const { get, set } = useUrlState()
  const meta = useApi<Z24Meta>('/api/z24/meta')
  const m = meta.data
  const week = get('week', m?.weeks[1]?.week ?? '')
  const thr = Number(get('rthr', '0.5'))
  const rung = get('rung', 'sched') as Rung
  const budget = Number(get('budget', '1'))
  const tab = get('tab', 'recognize') as Tab
  const w = useApi<Z24Week>('/api/z24/week', { week, thr }, !!week)

  if (meta.error) return <ErrorState error={meta.error} />
  if (!m) return <Loading />
  const d = w.data
  const tech = Object.fromEntries(m.techniques.map((t) => [t.id, t]))

  return (
    <>
      <Busy on={w.isFetching} />
      <PageHeader title="Attacker campaigns: recognise, then forecast the next burst"
        right={<Link to={`/dashboard/response?ds=z24&src=${week}&rung=${rung}&budget=${budget}`} className="btn btn-primary"><ShieldCheck className="size-4" /> Respond to alerts</Link>}>
        Behaviour in each host-minute → which ATT&amp;CK technique is running → when this attacker's next burst of it will start. Each week is scored by models trained without it.
      </PageHeader>

      <Toolbar>
        <Field label="Capture week">
          <Select value={week} onChange={(v) => set({ week: v, host: null, i: null, fi: null, day: null })}
            options={m.weeks.map((x) => ({ value: x.week, label: `${weekday(x.week)} · ${x.kind === 'attack' ? 'attack week' : 'benign week'}` }))} />
        </Field>
        <Field label="Recognizer threshold" hint="Fixed at 0.50 in all reported results">
          <div className="w-56"><Slider value={thr} min={0.1} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ rthr: v, i: null })} /></div>
        </Field>
        <Field label="Forecast model">
          <Segmented value={rung} onChange={(v) => set({ rung: v })} items={(['sched', 'world', 'renewal'] as Rung[]).map((r) => ({ id: r, label: RUNG_SHORT[r], title: m.rungs[r] }))} />
        </Field>
        <Field label="Alert budget">
          <Segmented value={budget} onChange={(v) => set({ budget: v })} items={m.budgets.map((b) => ({ id: b, label: `${b}/h`, title: `${b} false alarm(s) per attacker-hour` }))} />
        </Field>
      </Toolbar>
      {w.error && <ErrorState error={w.error} />}
      {!d && !w.error && <Loading label="Loading the week's flows, bucketing per host and scoring" />}

      {d && (
        <>
          <StatRow>
            <Stat label="Flows" value={int(d.n_flows)} />
            <Stat label="Host-minutes with traffic" value={int(d.traffic_minutes)} />
            <Stat label="Minutes flagged" value={int(d.flagged_any)} hint="Any technique" />
            {d.is_attack ? (
              <Stat label="Attacker sequences" value={int(d.attacker_sequences)} hint={`${d.attackers.length} attacker hosts, chosen from predicted flags`} />
            ) : (
              <Stat label="False alarms / host-hour" value={num(d.benign_false_alarms_per_host_hour, 3)} hint="Benign week: every flag is a false alarm" tone="good" />
            )}
            <Stat label="Week type" value={d.is_attack ? 'Attack' : 'Benign'} hint={d.is_attack ? 'Scripted campaign replay' : 'No attacks in the data'} />
          </StatRow>

          <Tabs value={tab} onChange={(v) => set({ tab: v })} items={[
            { id: 'recognize', label: 'Recognise techniques' },
            { id: 'forecast', label: 'Forecast attacker bursts', count: d.attackers.length || undefined },
            { id: 'structure', label: 'Campaign structure' },
            { id: 'evidence', label: 'Benchmark & evidence' },
          ]} />

          {tab === 'recognize' && <Recognize d={d} week={week} thr={thr} tech={tech} />}
          {tab === 'forecast' && (d.attackers.length ? <Forecast d={d} week={week} thr={thr} rung={rung} budget={budget} m={m} /> : <Card><Empty>No host sends technique-flagged traffic in this week, so there is no attacker to forecast.</Empty></Card>)}
          {tab === 'structure' && <Structure week={week} m={m} />}
          {tab === 'evidence' && <Evidence m={m} />}
        </>
      )}
    </>
  )
}

function Recognize({ d, week, thr, tech }: { d: Z24Week; week: string; thr: number; tech: Record<string, { name: string; tactic: string }> }) {
  const { get, set } = useUrlState()
  const host = get('host', d.flagged_hosts[0]?.host ?? '')
  const h = useApi<Z24RecogHost>('/api/z24/recognize/host', { week, thr, host }, !!host)
  const i = get('i') || (h.data?.top[0]?.i != null ? String(h.data.top[0].i) : '')
  const ex = useApi<Z24RecogExplain>('/api/z24/recognize/explain', { week, thr, i }, !!i)
  return (
    <div className="space-y-5">
      <Card title="Technique recognition this week" subtitle="Which ATT&CK technique each host-minute contains, from that minute's behaviour only">
        {!d.is_attack && <div className="mb-3"><Note>This is a benign week: every flagged minute is a false alarm.</Note></div>}
        <Table rows={d.techniques} rowKey={(r) => r.id} columns={[
          { key: 't', label: 'Technique', value: (r) => r.id, render: (r) => <span className="inline-flex items-center gap-2"><span className="size-2.5 rounded-[3px]" style={{ background: TECH_COLOR[r.id] }} /><span className="font-mono text-[12px] text-muted">{r.id}</span>{r.name}</span> },
          { key: 'tactic', label: 'ATT&CK tactic', value: (r) => r.tactic },
          { key: 'flagged', label: 'Flagged minutes', value: (r) => r.flagged, align: 'right', render: (r) => int(r.flagged) },
          ...(d.is_attack ? [
            { key: 'true', label: 'True minutes', value: (r: Z24Week['techniques'][number]) => r.true, align: 'right' as const, render: (r: Z24Week['techniques'][number]) => int(r.true) },
            { key: 'recall', label: 'Recall', value: (r: Z24Week['techniques'][number]) => r.recall, align: 'right' as const, render: (r: Z24Week['techniques'][number]) => <ScoreBar value={r.recall} format={(v) => v.toFixed(2)} /> },
            { key: 'precision', label: 'Precision', value: (r: Z24Week['techniques'][number]) => r.precision, align: 'right' as const, render: (r: Z24Week['techniques'][number]) => <ScoreBar value={r.precision} format={(v) => v.toFixed(2)} /> },
          ] : []),
        ]} />
      </Card>
      {d.flagged_hosts.length ? (
        <>
          <Card title={<span className="flex items-center gap-3">Host activity<Select value={host} onChange={(v) => set({ host: v, i: null })} className="h-8 font-mono text-[13px]" options={d.flagged_hosts.map((x) => ({ value: x.host, label: `${x.host} · ${x.minutes} flagged min` }))} /></span>}
            subtitle="Each tick is one host-minute. Upper lane: recognised by the model. Lower lane: ground truth (labelled weeks)."
            actions={<Legend items={[{ label: 'Recognised (technique colour)', color: 'var(--s1)' }, { label: 'Ground truth', color: 'var(--ink-2)' }]} />}>
            {h.data ? <TickLanes rows={h.data.ticks.map((t) => ({ label: tech[t.id].name, color: TECH_COLOR[t.id], a: t.recognized, b: t.truth }))} /> : <Loading />}
          </Card>
          {h.data && (
            <div className="grid gap-5 xl:grid-cols-[280px_minmax(0,1fr)]">
              <Card title="Most confident minutes" pad={false}>
                <ul className="max-h-[420px] overflow-y-auto py-1">
                  {h.data.top.map((x) => {
                    const j = x.p.indexOf(Math.max(...x.p))
                    return (
                      <li key={x.i}>
                        <button onClick={() => set({ i: x.i })} className={clsx('flex w-full items-center justify-between gap-2 px-5 py-2 text-left text-[13px] tnum', String(x.i) === i ? 'bg-accent-wash' : 'hover:bg-surface-hover')}>
                          <span>{stamp(x.t)}</span>
                          <span className="inline-flex items-center gap-1.5 text-ink-2"><span className="size-2 rounded-full" style={{ background: TECH_COLOR[Object.keys(TECH_COLOR)[j]] }} />{Object.keys(TECH_COLOR)[j]} {pct(x.p[j])}</span>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </Card>
              {ex.data ? (
                <div className="grid gap-5 lg:grid-cols-2">
                  <Card title="Technique probabilities" subtitle={stamp(ex.data.t)}>
                    <ProbBars items={ex.data.probs.map((p) => ({ label: p.name, sub: p.id, value: p.p, color: TECH_COLOR[p.id] }))} highlight={tech[ex.data.explained].name} />
                  </Card>
                  <Card title={`Why ${tech[ex.data.explained].name}`} subtitle="TreeSHAP, log-odds">
                    <DriverBars items={ex.data.drivers} />
                  </Card>
                </div>
              ) : <Card><Loading /></Card>}
            </div>
          )}
        </>
      ) : <Card><Empty>No minute was flagged in this week.</Empty></Card>}
    </div>
  )
}

function Forecast({ d, week, thr, rung, budget, m }: { d: Z24Week; week: string; thr: number; rung: Rung; budget: number; m: Z24Meta }) {
  const { get, set } = useUrlState()
  const host = get('atk', d.attackers[0].host)
  const tj = get('tech', m.techniques[0].id)
  const day = get('day')
  const f = useApi<Z24Forecast>('/api/z24/forecast', { week, thr, host, tech: tj, rung, budget, day })
  const fi = get('fi') || (f.data?.top[0] ? String(f.data.top[0].i) : '')
  const ex = useApi<{ drivers: { feature: string; value: number }[] }>('/api/z24/forecast/explain', { week, thr, tech: tj, i: fi }, !!fi && rung === 'sched')
  const tname = m.techniques.find((t) => t.id === tj)!.name
  const F = f.data
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-4">
        <Field label="Attacker"><Select value={host} onChange={(v) => set({ atk: v, day: null, fi: null })} className="font-mono" options={d.attackers.map((a) => ({ value: a.host, label: `${a.host} · ${a.days.length} day(s)` }))} /></Field>
        <Field label="Technique"><Select value={tj} onChange={(v) => set({ tech: v, fi: null })} options={m.techniques.map((t) => ({ value: t.id, label: `${t.id} ${t.name}` }))} /></Field>
        {F && <Field label="Day"><Select value={F.day} onChange={(v) => set({ day: v, fi: null })} options={F.days.map((x) => ({ value: x, label: weekday(x) }))} /></Field>}
      </div>
      {f.error && <ErrorState error={f.error} />}
      {F && (
        <>
          <StatRow>
            <Stat label="Bursts this week" value={int(F.stats.onsets)} hint={`${tname}, all attackers`} />
            <Stat label="Warned ≥1 min ahead" value={pct(F.stats.warned)} />
            <Stat label="False alarms / attacker-hour" value={num(F.stats.false_alarms_per_attacker_hour)} />
            <Stat label="Median lead when warned" value={F.stats.median_lead_min != null ? `${num(F.stats.median_lead_min, 0)} min` : '-'} />
            <Stat label="Alert threshold" value={num(F.threshold, 3)} hint={`${budget} false alarm(s)/attacker-hour on held-out minutes`} />
          </StatRow>
          <Card title={`P(${tname} burst in the next ${F.K} min)`} subtitle={`${host} · ${weekday(F.day)}. Click a minute to explain it.`}
            actions={<Legend items={[{ label: RUNG_SHORT[rung], color: 'var(--s1)', shape: 'line' }, { label: 'Renewal baseline', color: 'var(--neutral-series)', shape: 'line' }, { label: 'Actual burst', color: 'var(--critical)', shape: 'line' }, { label: 'Alert', color: 'var(--critical)', shape: 'dot' }]} />}>
            <TimeChart height={300} yFormat={(v) => pct(v)}
              series={[{ id: 'm', label: RUNG_SHORT[rung], color: 'var(--s1)', t: F.series.t, v: F.series.model }, { id: 'b', label: 'Renewal baseline', color: 'var(--neutral-series)', t: F.series.t, v: F.series.baseline }]}
              rules={F.bursts.map((t) => ({ t, color: 'var(--critical)' }))} points={F.alerts.t.map((t, k) => ({ t, v: F.alerts.p[k], color: 'var(--critical)' }))}
              threshold={{ v: F.threshold, label: `threshold ${F.threshold.toFixed(2)}` }}
              selected={fi ? F.series.t[F.series.i.indexOf(Number(fi))] ?? null : null}
              onPick={(_, j) => set({ fi: F.series.i[j] })} />
            <p className="mt-2 text-[12px] text-muted">An alert followed by a burst within a few minutes is early warning.</p>
          </Card>
          <div className="grid gap-5 xl:grid-cols-[280px_minmax(0,1fr)]">
            <Card title="Highest-risk minutes" pad={false}>
              <ul className="max-h-[380px] overflow-y-auto py-1">
                {F.top.map((x) => (
                  <li key={x.i}>
                    <button onClick={() => set({ fi: x.i })} className={clsx('flex w-full items-center justify-between px-5 py-2 text-[13px] tnum', String(x.i) === fi ? 'bg-accent-wash' : 'hover:bg-surface-hover')}>
                      <span>{stamp(x.t)}</span><span className={clsx(x.p >= F.threshold ? 'font-medium text-critical' : 'text-ink-2')}>{pct(x.p)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
            <Card title="Why this forecast" subtitle="TreeSHAP for the history GBDT. 'since' = minutes since the last recognised burst; 'n60'/'n150' = flagged minutes in the last 60/150 min.">
              {rung !== 'sched' ? <Note>Explanations are shown for the History GBDT model. Switch the forecast model to see them.</Note> : ex.data ? <DriverBars items={ex.data.drivers} /> : <Loading />}
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

function Structure({ week, m }: { week: string; m: Z24Meta }) {
  const s = useApi<Z24Structure>('/api/z24/structure', { week })
  if (s.error) return <ErrorState error={s.error} />
  if (!s.data) return <Loading />
  const S = s.data
  const name = (id: string) => m.techniques.find((t) => t.id === id)!.name
  const cols = Object.keys(S.attackers[0] ?? {})
  return (
    <div className="space-y-5">
      {S.is_attack ? (
        <>
          <Card title="When in the hour does each burst start?" subtitle="First flow of each technique in each clock hour, per attacker. Spread evenly: the bursts are not tied to a clock time.">
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{S.minute_hist.map((h) => <Histogram key={h.id} title={name(h.id)} counts={h.counts} step={5} color={TECH_COLOR[h.id]} xLabel="minute" />)}</div>
          </Card>
          <Card title="Minutes between consecutive bursts" subtitle="Each technique repeats roughly hourly with large jitter, so time since the last burst is informative but far from decisive.">
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{S.gap_hist.map((h) => <Histogram key={h.id} title={name(h.id)} counts={h.counts} step={5} color={TECH_COLOR[h.id]} xLabel="gap (min)" />)}</div>
          </Card>
        </>
      ) : <Note>Burst-timing histograms need an attack week. The table below covers all attack weeks.</Note>}
      <Card title="Does one technique lead to another?" subtitle={`P(technique in row starts within 5 min after a burst of the technique in column), all attack weeks. A random hourly schedule gives about 0.08.`}>
        <div className="overflow-x-auto">
          <table className="text-[13px]">
            <thead><tr><th />{S.lags.techs.map((t) => <th key={t} className="px-2 pb-2 text-left text-[11.5px] font-semibold text-muted">after {name(t)}</th>)}</tr></thead>
            <tbody>
              {S.lags.matrix.map((row, r) => (
                <tr key={r}>
                  <td className="pr-4 text-ink-2">{name(S.lags.techs[r])}</td>
                  {row.map((v, c) => (
                    <td key={c} className="p-0.5">
                      <div className={clsx('flex h-10 w-32 items-center justify-center rounded-md text-[13px] font-medium tnum', v != null && v > 0.3 / 1 ? 'text-white' : 'text-ink')} style={{ background: v == null ? 'var(--surface-2)' : seq(v / 0.3) }}>
                        {v == null ? '-' : v.toFixed(2)}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[12.5px] text-ink-2">Values of 0.10-0.13 mean weak coupling: there is no recon → exploit → credential-access chain to learn. Minute-of-hour spread: std {num(S.minute_std, 1)} min vs {num(S.uniform_std, 1)} for a perfectly uniform schedule.</p>
      </Card>
      <Card title="Attackers across all weeks" subtitle="Flagged minutes per technique">
        <Table<Record<string, string | number>> dense maxHeight={360} rows={S.attackers} rowKey={(r) => `${r.host}${r.week}`} initialSort={{ key: 'week', dir: 'asc' }}
          columns={cols.map((c) => ({ key: c, label: c, value: (r) => r[c], align: (typeof S.attackers[0][c] === 'number' ? 'right' : 'left') as 'right' | 'left', render: (r) => (c === 'host' ? <span className="font-mono">{r[c]}</span> : typeof r[c] === 'number' ? int(r[c] as number) : String(r[c])) }))} />
      </Card>
    </div>
  )
}

type Row = Record<string, number>
interface ZMetrics {
  dataset: { attacker_hosts: number; attacker_sequences: number }
  recognizer: { per_technique: Record<string, { pr_auc: number; n_pos: number; 'at_0.5': Record<string, number> }>; any_attack_detector: Record<string, number>; 'confusion@0.5': Record<string, Record<string, number>> }
  forecast: { macro: Record<string, Row>; per_technique: Record<string, Record<string, Row>> }
}
const LAB: Record<string, string> = { renewal: 'Renewal hazard (time since last burst)', state: 'Current behaviour only', own: 'Own-technique history', sched: 'All-technique history (GBDT)', sched_shuf: '… history order shuffled', world: 'World model (GRU rollout + head)', world_shuf: '… history order shuffled (blocks)', sched_oracle: 'All-technique history, true flags (upper bound)', clock: 'Ablation: + wall-clock minute (not a valid model)' }

function Evidence({ m }: { m: Z24Meta }) {
  const q = useApi<ZMetrics>('/api/z24/metrics')
  const name = (id: string) => m.techniques.find((t) => t.id === id)?.name ?? id
  const data = q.data
  const rows = useMemo(() => (data ? Object.keys(LAB).filter((k) => data.forecast.macro[k]).map((k) => ({ k, v: data.forecast.macro[k] })) : []), [data])
  if (q.error) return <ErrorState error={q.error} />
  if (!data) return <Loading />
  const rec = data.recognizer, fc = data.forecast.macro, ds = data.dataset
  const Q = (k: string) => fc[k].pr_auc
  const base = fc.sched.prevalence
  const prs = Object.values(rec.per_technique).map((v) => v.pr_auc)
  const best = Math.max(...rows.filter((r) => r.k !== 'sched_oracle' && r.k !== 'clock').map((r) => r.v.pr_auc))
  return (
    <div className="space-y-5">
      <Card title="What this evidence does and does not support">
        <Findings items={[
          { title: 'Recognition is easy here, and that is a warning, not a win.', body: <>Techniques separate at PR-AUC {num(Math.min(...prs))}-{num(Math.max(...prs))} in a held-out week of the same scripted campaign; the scripts are stereotyped and attack minutes contain no benign traffic.</> },
          { title: 'Forecasting the next burst is genuinely temporal.', body: <>History models reach macro PR-AUC {num(Q('sched'))} against a base rate of {num(base, 3)} ({num(Q('sched') / base, 1)}×); current behaviour alone gives {num(Q('state'))}. Shuffling history order costs {pct((Q('sched') - Q('sched_shuf')) / Q('sched'))} (GBDT) and {pct((Q('world') - Q('world_shuf')) / Q('world'))} (world model).</> },
          { title: 'The world model matches, not beats, hand-built history features:', body: <>{num(Q('world'), 3)} vs {num(Q('sched'), 3)}.</> },
          { title: 'The signal is modest:', body: <>recall {pct(fc.sched['recall@1.0/h'])} at one false alarm per attacker-hour. Adding the wall clock (an ablation, not allowed in shipped models) reaches only {num(Q('clock'))}.</> },
          { title: 'Not a kill chain.', body: <>Other techniques' history adds {pct((Q('sched') - Q('own')) / Q('own'))} over own-technique history, and cross-technique start lags are near random.</> },
          { title: 'Scope:', body: <>one lab, one campaign replayed for 5 weeks, {ds.attacker_hosts} attacker hosts. This measures repeatability of a schedule, not generalisation to unseen attacks.</> },
        ]} />
      </Card>
      <Card title="Task A · recognise the technique from one host-minute" subtitle="Leave-one-attack-week-out">
        <Table rows={Object.entries(rec.per_technique)} rowKey={([t]) => t} columns={[
          { key: 't', label: 'Technique', render: ([t]) => <span className="inline-flex items-center gap-2"><span className="size-2.5 rounded-[3px]" style={{ background: TECH_COLOR[t] }} />{name(t)}</span> },
          { key: 'pr', label: 'PR-AUC', align: 'right', value: ([, v]) => v.pr_auc, render: ([, v]) => num(v.pr_auc, 3) },
          { key: 'rec', label: 'Recall @0.5', align: 'right', render: ([, v]) => num(v['at_0.5'].recall, 3) },
          { key: 'prec', label: 'Precision @0.5', align: 'right', render: ([, v]) => num(v['at_0.5'].precision, 3) },
          { key: 'ben', label: 'Benign alarms / host-h', align: 'right', render: ([, v]) => num(v['at_0.5'].benign_alarms_per_host_hour, 4) },
          { key: 'oth', label: "Fires on other techniques' minutes", align: 'right', render: ([, v]) => num(v['at_0.5'].other_technique_false_fire_rate, 3) },
          { key: 'n', label: 'Positive minutes', align: 'right', value: ([, v]) => v.n_pos, render: ([, v]) => int(v.n_pos) },
        ]} />
      </Card>
      <Card title="Task B · forecast the next burst (macro over 5 techniques)" subtitle={`Population: minutes where the technique is not active; positive = a burst starts within 5 min. Base rate ${num(base, 3)}. @1/h = 1 false alarm per attacker-hour.`}>
        <Table rows={rows} rowKey={(r) => r.k} columns={[
          { key: 'm', label: 'Model', render: (r) => <span className={clsx(r.k.includes('shuf') && 'pl-3 text-muted', (r.k === 'clock' || r.k === 'sched_oracle') && 'text-muted italic')}>{LAB[r.k]}</span> },
          { key: 'pr', label: 'PR-AUC', align: 'right', value: (r) => r.v.pr_auc, render: (r) => <ScoreBar value={r.v.pr_auc} best={r.v.pr_auc === best} format={(v) => v.toFixed(3)} /> },
          { key: 'roc', label: 'ROC-AUC', align: 'right', value: (r) => r.v.roc_auc, render: (r) => num(r.v.roc_auc, 3) },
          { key: 'rec', label: 'Recall @1/h', align: 'right', render: (r) => num(r.v['recall@1.0/h'], 3) },
          { key: 'prec', label: 'Precision @1/h', align: 'right', render: (r) => num(r.v['precision@1.0/h'], 3) },
          { key: 'on', label: 'Onsets warned', align: 'right', render: (r) => pct(r.v['onset_detected@1.0/h']) },
          { key: 'brier', label: 'Brier', align: 'right', render: (r) => num(r.v.brier, 3) },
        ]} />
      </Card>
      <Card title="PR-AUC per technique">
        <Table rows={['renewal', 'state', 'sched', 'world', 'sched_shuf']} rowKey={(k) => k} columns={[
          { key: 'm', label: 'Model', render: (k) => LAB[k] },
          ...m.techniques.map((t) => ({ key: t.id, label: t.name, align: 'right' as const, render: (k: string) => num(data.forecast.per_technique[k][t.id]?.pr_auc, 3) })),
        ]} />
      </Card>
      <Badge>Numbers from results/z24/metrics.json</Badge>
    </div>
  )
}
