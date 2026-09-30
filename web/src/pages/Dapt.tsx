import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { ShieldCheck, UploadSimple as Upload } from '@phosphor-icons/react'
import { useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { DriverBars, ProbBars } from '../components/charts/Bars'
import { Heatmap, RampLegend } from '../components/charts/Heatmap'
import { RolloutChart } from '../components/charts/Small'
import { TimeChart, toBands } from '../components/charts/TimeChart'
import { Table, type Column } from '../components/Table'
import { Badge, Busy, Card, Empty, ErrorState, Field, Findings, Legend, Loading, Note, PageHeader, Segmented, Select, Slider, Stat, StatRow, Tabs, Toolbar } from '../components/ui'
import { uploadFlows, useApi, type DaptExplain, type DaptMeta, type DaptModel, type DaptScores, type DaptSummary } from '../lib/api'
import { STAGE_COLOR } from '../lib/colors'
import { int, num, pct, stamp, weekday } from '../lib/format'
import { useUrlState } from '../lib/url'

type Tab = 'overview' | 'investigate' | 'evidence' | 'method'
const MODEL_LABEL: Record<DaptModel, string> = { lag: 'LightGBM + lags', world: 'World model (GRU rollout)' }

export default function DaptPage() {
  const { get, set } = useUrlState()
  const meta = useApi<DaptMeta>('/api/dapt/meta')
  const m = meta.data
  const source = get('src', m?.days[1] ?? '')
  const model = get('model', m?.primary ?? 'lag') as DaptModel
  const thr = Number(get('thr', m ? String(m.thresholds[model]) : '0.2'))
  const tab = get('tab', 'overview') as Tab
  const scores = useApi<DaptScores>('/api/dapt/scores', { source }, !!source)
  const summary = useApi<DaptSummary>('/api/dapt/summary', { source, model, thr }, !!source && !!m)

  const fileRef = useRef<HTMLInputElement>(null)
  const qc = useQueryClient()
  const [uploading, setUploading] = useState(false)
  const [upErr, setUpErr] = useState<string | null>(null)
  const onFile = async (f: File | undefined) => {
    if (!f) return
    setUploading(true)
    setUpErr(null)
    try {
      const r = await uploadFlows(f)
      await qc.invalidateQueries({ queryKey: ['/api/dapt/meta'] })
      set({ src: r.source, host: null, i: null })
    } catch (e) {
      setUpErr((e as Error).message)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  if (meta.error) return <ErrorState error={meta.error} />
  if (!m) return <Loading />
  const isUpload = source.startsWith('upload:')
  const s = summary.data
  const sc = scores.data

  return (
    <>
      <Busy on={scores.isFetching || summary.isFetching || uploading} />
      <PageHeader title="Forecasting attacks before they start"
        right={<Link to={`/dashboard/response?ds=dapt&src=${isUpload ? m.days[1] : source}&model=${model}&thr=${thr}`} className="btn btn-primary"><ShieldCheck className="size-4" /> Respond to alerts</Link>}>
        Every monitored host gets a state vector per minute. The model forecasts P(attack traffic in the next {m.K} minutes), maps it to a kill-chain stage and explains why.
      </PageHeader>

      <Toolbar>
        <Field label="Traffic source" hint={isUpload ? 'Scored by the model trained on all five days (in-sample for DAPT files).' : 'Scored by a model trained on the other four days.'}>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={source} onChange={(v) => set({ src: v, host: null, i: null })}
              options={[...m.days.map((d) => ({ value: d, label: `${weekday(d)} · ${m.onsets_by_day[d]} attack onsets` })), ...m.uploads.map((u) => ({ value: u.source, label: `Upload · ${u.name}` }))]} />
            <button onClick={() => fileRef.current?.click()} className="btn btn-quiet">
              <Upload className="size-3.5" /> {uploading ? 'Scoring…' : 'Upload CSV'}
            </button>
            <input ref={fileRef} type="file" accept=".csv,.gz" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          </div>
        </Field>
        <Field label="Risk model">
          <Segmented value={model} onChange={(v) => set({ model: v, thr: null })}
            items={(['lag', 'world'] as DaptModel[]).map((k) => ({ id: k, label: <>{MODEL_LABEL[k]}{k === m.primary && <span className="ml-1.5 text-[10.5px] text-accent-ink">best</span>}</> }))} />
        </Field>
        <Field label="Alert threshold">
          <div className="flex w-64 items-center gap-2">
            <Slider value={thr} min={0.01} max={0.99} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set({ thr: v })} />
            {Math.abs(thr - m.thresholds[model]) > 1e-6 && (
              <button onClick={() => set({ thr: null })} className="shrink-0 text-[12px] text-accent-ink hover:underline" title="1 false alarm per host-hour on held-out benign minutes">reset</button>
            )}
          </div>
        </Field>
      </Toolbar>
      {upErr && <ErrorState error={new Error(upErr)} />}
      {(scores.error || summary.error) && <ErrorState error={scores.error ?? summary.error} />}

      {s && (
        <StatRow>
          <Stat label="Flows analysed" value={int(s.n_flows)} hint={`${s.n_hosts} monitored hosts · ${int(s.host_minutes)} host-minutes`} />
          <Stat label="Alert minutes" value={int(s.alert_minutes)} hint={`at threshold ${thr.toFixed(2)}`} />
          {s.labelled ? (
            <>
              <Stat label="False alarms / host-hour" value={num(s.false_alarms_per_host_hour)} hint="Alerts with no attack in the next 5 min" />
              <Stat label="Onsets warned ≥1 min ahead" value={s.onsets ? `${s.onsets_warned} / ${s.onsets}` : '-'} hint={s.onsets ? `${pct(s.onsets_warned! / s.onsets)} of attack onsets` : 'No attack on this day'} />
              <Stat label="Median warning lead" value={s.median_lead_min != null ? `${num(s.median_lead_min, 0)} min` : '-'} hint="When warned" />
            </>
          ) : (
            <Stat label="Ground truth" value="-" hint="No labels in this file" />
          )}
        </StatRow>
      )}

      <Tabs value={tab} onChange={(v) => set({ tab: v })}
        items={[{ id: 'overview', label: 'Network' }, { id: 'investigate', label: 'Investigate a host' }, { id: 'evidence', label: 'Benchmark & evidence' }, { id: 'method', label: 'How it works' }]} />

      {!sc && !scores.error && tab !== 'evidence' && tab !== 'method' && <Loading label="Bucketing flows per host and scoring" />}
      {sc && s && tab === 'overview' && <Overview sc={sc} s={s} model={model} thr={thr} m={m} onPick={(host, i) => set({ tab: 'investigate', host, i })} />}
      {sc && tab === 'investigate' && <Investigate sc={sc} model={model} thr={thr} m={m} source={source} />}
      {tab === 'evidence' && <Evidence />}
      {tab === 'method' && <Method m={m} />}
    </>
  )
}

interface HostRow { host: string; peak: number; alerts: number; attack: number }

function useHostRank(sc: DaptScores, model: DaptModel, thr: number) {
  return useMemo(() => {
    const r = sc.rows
    const acc = sc.hosts.map((h) => ({ host: h, peak: 0, alerts: 0, attack: 0 }))
    for (let k = 0; k < r.host.length; k++) {
      const a = acc[r.host[k]], p = r[model][k]
      if (p > a.peak) a.peak = p
      if (p >= thr) a.alerts++
      a.attack += r.attack[k]
    }
    return acc.sort((a, b) => b.peak - a.peak)
  }, [sc, model, thr])
}

function Overview({ sc, s, model, thr, m, onPick }: { sc: DaptScores; s: DaptSummary; model: DaptModel; thr: number; m: DaptMeta; onPick: (host: string, i?: number) => void }) {
  const rank = useHostRank(sc, model, thr)
  const order = rank.map((r) => r.host)
  const pos = new Map(order.map((h, k) => [sc.hosts.indexOf(h), k]))
  const rowOf = sc.rows.host.map((h) => pos.get(h)!)
  const vmax = Math.max(0.05, ...sc.rows[model])
  const cols: Column<HostRow>[] = [
    { key: 'host', label: 'Host', value: (r) => r.host, render: (r) => <span className="font-mono">{r.host}</span> },
    { key: 'peak', label: 'Peak risk', value: (r) => r.peak, align: 'right', render: (r) => <span className={clsx(r.peak >= thr && 'font-semibold text-critical')}>{pct(r.peak)}</span> },
    { key: 'alerts', label: 'Alert minutes', value: (r) => r.alerts, align: 'right' },
    ...(sc.labelled ? [{ key: 'attack', label: 'True attack minutes', value: (r: HostRow) => r.attack, align: 'right' as const }] : []),
  ]
  const findRow = (host: string, t: number) => {
    const hi = sc.hosts.indexOf(host)
    return sc.rows.t.findIndex((tt, k) => sc.rows.host[k] === hi && tt === t - 60_000 * 1)
  }
  return (
    <div className="space-y-5">
      <Card title="Network risk" subtitle={`Forecast risk per host and minute (${MODEL_LABEL[model]}). Click a host to investigate.`}
        actions={<div className="flex flex-wrap items-center gap-4"><RampLegend label="Risk" lo="0" hi={pct(vmax)} />{sc.labelled && <Legend items={Object.entries(STAGE_COLOR).map(([k, c]) => ({ label: k, color: c }))} />}</div>}>
        <Heatmap rows={order} host={rowOf} t={sc.rows.t} v={sc.rows[model]} max={vmax} valueLabel="Risk" format={(v) => pct(v, 1)}
          marker={sc.labelled ? sc.rows.stage : undefined} markerColor={(k) => STAGE_COLOR[m.stages[k]]} markerLabel={(k) => m.stages[k]} onRowClick={(h) => onPick(h)} />
        {sc.labelled && <p className="mt-2 text-[12px] text-muted">The coloured strip under each row marks true attack minutes by stage. Risk that lights up before a strip is early warning.</p>}
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Hosts" subtitle="Ranked by peak forecast risk">
          <Table rows={rank} columns={cols} rowKey={(r) => r.host} onRowClick={(r) => onPick(r.host)} initialSort={{ key: 'peak', dir: 'desc' }} />
        </Card>
        {s.labelled && (
          <Card title="Attack onsets" subtitle="Each time a benign host starts attack traffic, and whether it was warned">
            {s.onset_list?.length ? (
              <Table rows={s.onset_list} rowKey={(r) => `${r.host}${r.t}`} maxHeight={420} dense
                onRowClick={(r) => { const i = findRow(r.host, r.t); onPick(r.host, i >= 0 ? i : undefined) }}
                columns={[
                  { key: 't', label: 'Onset', value: (r) => r.t, render: (r) => stamp(r.t) },
                  { key: 'host', label: 'Host', value: (r) => r.host, render: (r) => <span className="font-mono">{r.host}</span> },
                  { key: 'stage', label: 'Stage', value: (r) => r.stage, render: (r) => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: STAGE_COLOR[r.stage] }} />{r.stage}</span> },
                  { key: 'lead', label: 'Warning', value: (r) => r.lead_min, align: 'right', render: (r) => (r.lead_min > 0 ? <Badge tone="good">{r.lead_min} min ahead</Badge> : <Badge>missed</Badge>) },
                ]} />
            ) : <Empty>No attack onsets on this day. Every alert here is a false alarm.</Empty>}
          </Card>
        )}
      </div>
    </div>
  )
}

function Investigate({ sc, model, thr, m, source }: { sc: DaptScores; model: DaptModel; thr: number; m: DaptMeta; source: string }) {
  const { get, set } = useUrlState()
  const rank = useHostRank(sc, model, thr)
  const host = get('host', rank[0]?.host)
  const hi = sc.hosts.indexOf(host)
  const idx = useMemo(() => sc.rows.host.flatMap((h, k) => (h === hi ? [k] : [])), [sc, hi])
  const top = useMemo(() => [...idx].sort((a, b) => sc.rows[model][b] - sc.rows[model][a]).slice(0, 25), [idx, sc, model])
  const iParam = get('i')
  const i = iParam !== '' && idx.includes(Number(iParam)) ? Number(iParam) : top[0]
  const t = idx.map((k) => sc.rows.t[k])
  const p = idx.map((k) => sc.rows[model][k])
  const bands = sc.labelled ? toBands(t, idx.map((k) => sc.rows.stage[k]), (c) => STAGE_COLOR[m.stages[c as number]], (c) => m.stages[c as number]) : []
  const alerts = idx.filter((k) => sc.rows[model][k] >= thr).map((k) => ({ t: sc.rows.t[k], v: sc.rows[model][k], color: 'var(--critical)' }))

  return (
    <div className="space-y-5">
      <Card title={<span className="flex items-center gap-3">Risk timeline<Select value={host} onChange={(h) => set({ host: h, i: null })} className="h-8 font-mono text-[13px]" options={rank.map((r) => ({ value: r.host, label: `${r.host} · peak ${pct(r.peak)}` }))} /></span>}
        subtitle={`P(attack in the next ${m.K} min) each minute. Click any minute to explain it.`}
        actions={<Legend items={[{ label: 'Forecast risk', color: 'var(--s1)', shape: 'line' }, { label: 'Alert', color: 'var(--critical)', shape: 'dot' }, { label: 'Threshold', color: 'var(--ink-2)', shape: 'dash' }, ...(sc.labelled ? Object.entries(STAGE_COLOR).map(([k, c]) => ({ label: k, color: c })) : [])]} />}>
        <TimeChart height={300} series={[{ id: 'p', label: 'Risk', color: 'var(--s1)', t, v: p }]} bands={bands} points={alerts} threshold={{ v: thr, label: `threshold ${thr.toFixed(2)}` }}
          yFormat={(v) => pct(v)} yLabel="risk" selected={i != null ? sc.rows.t[i] : null} onPick={(_, j) => set({ i: idx[j] })} />
      </Card>
      <div className="grid gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
        <Card title="Highest-risk minutes" pad={false} className="xl:sticky xl:top-6 xl:self-start">
          <ul className="max-h-[560px] overflow-y-auto py-1">
            {top.map((k) => (
              <li key={k}>
                <button onClick={() => set({ i: k })} className={clsx('flex w-full items-center justify-between gap-2 px-5 py-2 text-left text-[13px] tnum', k === i ? 'bg-accent-wash' : 'hover:bg-surface-hover')}>
                  <span className="flex items-center gap-2">{stamp(sc.rows.t[k])}{sc.labelled && sc.rows.attack[k] > 0 && <span className="size-1.5 rounded-full bg-critical" title="attack in progress" />}</span>
                  <span className={clsx('font-medium', sc.rows[model][k] >= thr ? 'text-critical' : 'text-ink-2')}>{pct(sc.rows[model][k])}</span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
        {i != null ? <Explain source={source} i={i} model={model} K={m.K} /> : <Card><Empty>Pick a minute.</Empty></Card>}
      </div>
    </div>
  )
}

const OUTCOME: Record<string, { label: string; tone: 'critical' | 'good' | 'neutral' | 'warning' }> = {
  attack_now: { label: 'Attack already in progress this minute', tone: 'critical' },
  attack_next: { label: 'An attack started within the horizon', tone: 'critical' },
  quiet: { label: 'No attack followed within the horizon', tone: 'good' },
  past_end: { label: 'Horizon runs past the end of the capture', tone: 'neutral' },
}

function Explain({ source, i, model, K }: { source: string; i: number; model: DaptModel; K: number }) {
  const e = useApi<DaptExplain>('/api/dapt/explain', { source, i, model })
  if (e.error) return <ErrorState error={e.error} />
  if (!e.data) return <Card><Loading label="Explaining" /></Card>
  const d = e.data
  const best = d.stage_probs.reduce((a, b) => (b.p > a.p ? b : a))
  return (
    <div className="min-w-0 space-y-5">
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Forecast" subtitle={`${d.host} · ${stamp(d.t)}`}>
          <div className="flex items-baseline gap-2">
            <span className="text-[40px] leading-none font-semibold tracking-tight tnum">{pct(d.p)}</span>
            <span className="text-[13px] text-ink-2">chance of attack traffic in the next {K} min</span>
          </div>
          {d.outcome && <div className="mt-3"><Badge tone={OUTCOME[d.outcome].tone}>What happened: {OUTCOME[d.outcome].label}</Badge></div>}
          <h4 className="mt-5 mb-2.5 text-[13px] font-semibold">Predicted stage if an attack starts</h4>
          <ProbBars items={d.stage_probs.map((x) => ({ label: x.stage, value: x.p, color: STAGE_COLOR[x.stage] }))} highlight={best.stage} />
          <p className="mt-3 text-[12px] text-muted">
            Read with care: across held-out days this stage model scored {pct(d.stage_accuracy)} (majority baseline {pct(d.stage_majority)}). It only works when the attack type was seen in training.
          </p>
        </Card>
        <Card title="Why this risk" subtitle="Exact TreeSHAP contributions, in log-odds">
          <DriverBars items={d.drivers} />
        </Card>
      </div>
      <Card title="World-model forward simulation" subtitle={`A GRU trained to predict the next host state is fed its own output for ${K} steps. Solid: simulated. Dashed: what was actually observed.`}
        actions={<Legend items={[{ label: 'Simulated', color: 'var(--s1)', shape: 'line' }, { label: 'Observed', color: 'var(--ink-2)', shape: 'dash' }]} />}>
        <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-4">
          {d.rollout.map((r) => <RolloutChart key={r.feature} title={r.feature} forecast={r.forecast} observed={r.observed} />)}
        </div>
      </Card>
      <Card title="Flows around this minute" subtitle={`${d.flows.rows.length} flows touching ${d.host} in the next ${K + 1} minutes${d.flows.rows.length >= 200 ? ' (first 200)' : ''}`}>
        {d.flows.rows.length ? (
          <Table<DaptExplain['flows']['rows'][number]> dense maxHeight={320} rows={d.flows.rows} rowKey={(_, k) => String(k)}
            columns={d.flows.columns.map((c, j) => ({
              key: c, label: c, value: (r) => r[j] as number | string,
              align: (typeof d.flows.rows[0][j] === 'number' && c !== 'ts' ? 'right' : 'left') as 'right' | 'left',
              render: (r) => (c === 'ts' ? stamp(r[j] as number) : c.endsWith('IP') ? <span className="font-mono">{String(r[j])}</span> : typeof r[j] === 'number' ? int(r[j] as number) : String(r[j] ?? '-')),
            }))} />
        ) : <Empty>No flows in this window.</Empty>}
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------- evidence (all numbers from results/demo/metrics.json)
type Pool = Record<string, Record<string, number>>
interface DMetrics {
  dataset: Record<string, number | string | string[] | Record<string, number>>
  pooled: Pool
  bootstrap: Record<string, { pr_auc_ci90: [number, number] }>
  per_day: Record<string, Record<string, Record<string, number>>>
  stage: { accuracy: number; majority_baseline_accuracy: number; n: number }
  secondary_blocked: { split: string; pooled: Pool; stage: { accuracy: number; majority_baseline_accuracy: number; n: number } }
}
const NAMES: Record<string, string> = { lr: 'Logistic regression', cur: 'LightGBM (current minute)', lag: 'LightGBM (+ lags)', world: 'World model (GRU rollout + head)', lag_shuf: 'LightGBM (+ lags), history shuffled', world_shuf: 'World model, history shuffled' }

function ModelTable({ pool, ci }: { pool: Pool; ci?: DMetrics['bootstrap'] }) {
  const rows = Object.keys(NAMES).filter((k) => pool[k]).map((k) => ({ k, v: pool[k] }))
  const bestPr = Math.max(...rows.map((r) => r.v.pr_auc ?? 0))
  type R = (typeof rows)[number]
  const c = (key: string, label: string, f: (v: number) => string, title?: string): Column<R> => ({ key, label, title, align: 'right', value: (r) => r.v[key], render: (r) => f(r.v[key]) })
  return (
    <Table rows={rows} rowKey={(r) => r.k} columns={[
      { key: 'model', label: 'Model', value: (r) => NAMES[r.k], render: (r) => <span className={clsx(r.k.endsWith('shuf') && 'text-muted')}>{NAMES[r.k]}</span> },
      { key: 'pr_auc', label: 'PR-AUC', align: 'right', value: (r) => r.v.pr_auc, render: (r) => <span className={clsx(r.v.pr_auc === bestPr && 'font-semibold text-ink')}>{num(r.v.pr_auc, 3)}</span> },
      ...(ci ? [{ key: 'ci', label: '90% CI', align: 'right' as const, render: (r: R) => ci[r.k] ? `${num(ci[r.k].pr_auc_ci90[0])}-${num(ci[r.k].pr_auc_ci90[1])}` : '-' }] : []),
      c('roc_auc', 'ROC-AUC', (v) => num(v, 3)),
      c('recall@1.0/h', 'Recall', (v) => num(v, 3), 'at 1 false alarm per host-hour'),
      c('precision@1.0/h', 'Precision', (v) => num(v, 3), 'at 1 false alarm per host-hour'),
      c('onset_detected@1.0/h', 'Onsets warned', (v) => pct(v)),
      c('median_lead_min@1.0/h', 'Median lead', (v) => (v == null ? '-' : `${num(v, 1)} min`)),
      c('brier', 'Brier', (v) => num(v, 3)),
      c('ece', 'ECE', (v) => num(v, 3)),
    ]} />
  )
}

function Evidence() {
  const q = useApi<DMetrics>('/api/dapt/metrics')
  if (q.error) return <ErrorState error={q.error} />
  if (!q.data) return <Loading />
  const mx = q.data, P = mx.pooled, ds = mx.dataset as Record<string, number>
  const prev = P.lr.prevalence
  const [lo, hi] = mx.bootstrap.lag.pr_auc_ci90
  const shuf = (a: string, s: string) => (P[s].pr_auc - P[a].pr_auc) / P[a].pr_auc
  const days = Object.keys(mx.per_day.lag)
  const sb = mx.secondary_blocked
  return (
    <div className="space-y-5">
      <Card title="What this evidence does and does not support">
        <Findings items={[
          { title: 'Ranking works; thresholds do not transfer.', body: <>Forecast PR-AUC {num(P.lag.pr_auc)} vs base rate {num(prev)} pooled ({num(P.lag.pr_auc / prev, 1)}× lift); within a single held-out day ROC-AUC is 0.74-0.95. Pooled alarm-budget recall is only {pct(P.lag['recall@1.0/h'])} because each day's attack type shifts the score scale.</> },
          { title: 'World model vs GBDT with lags:', body: <>{num(P.world.pr_auc)} vs {num(P.lag.pr_auc)} PR-AUC (lag 90% CI {num(lo)}-{num(hi)}). {P.world.pr_auc <= hi ? 'No measurable improvement on this data.' : 'Above the lag CI.'}</> },
          { title: 'Shuffle control:', body: <>shuffling the order of the history minutes changes PR-AUC by {pct(shuf('lag', 'lag_shuf'))} (lag) and {pct(shuf('world', 'world_shuf'))} (world). {Math.min(shuf('lag', 'lag_shuf'), shuf('world', 'world_shuf')) > -0.1 ? 'Neither model loses skill, so here the forecast comes from the current state, not learned temporal dynamics.' : 'At least one model relies on temporal order.'}</> },
          { title: 'Stage forecasting is at chance across days', body: <>({pct(mx.stage.accuracy)} vs {pct(mx.stage.majority_baseline_accuracy)} majority) because each day contains a different stage; it reaches {pct(sb.stage.accuracy)} vs {pct(sb.stage.majority_baseline_accuracy)} when stages are shared.</> },
          { title: 'Scope:', body: <>{ds.internal_hosts} monitored hosts, {ds.attack_buckets} attack minutes, one lab, one week. The pipeline is the deliverable; the numbers are what this small corpus supports.</> },
        ]} />
      </Card>
      <Card title="Primary: leave-one-day-out" subtitle={`Each day forecast by models that never saw it. Base rate (random scorer's PR-AUC) = ${num(prev, 3)}. ${ds.onset_episodes} onset episodes on ${ds.host_days} host-days. Operating point: 1 false alarm per host-hour.`}>
        <ModelTable pool={P} ci={mx.bootstrap} />
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Within-day ranking (ROC-AUC per held-out day)" subtitle="Days with no attacks are blank">
          <Table rows={['lr', 'cur', 'lag', 'world']} rowKey={(k) => k} columns={[
            { key: 'm', label: 'Model', render: (k) => NAMES[k] },
            ...days.map((d) => ({ key: d, label: weekday(d), align: 'right' as const, render: (k: string) => num(mx.per_day[k][d]?.roc_auc) })),
          ]} />
        </Card>
        <Card title="Stage forecast accuracy">
          <Table rows={[
            { k: 'Accuracy', a: pct(mx.stage.accuracy), b: pct(sb.stage.accuracy) },
            { k: 'Majority-class baseline', a: pct(mx.stage.majority_baseline_accuracy), b: pct(sb.stage.majority_baseline_accuracy) },
            { k: 'Onset minutes evaluated', a: int(mx.stage.n), b: int(sb.stage.n) },
          ]} rowKey={(r) => r.k} columns={[{ key: 'k', label: '', render: (r) => r.k }, { key: 'a', label: 'Leave-one-day-out', align: 'right', render: (r) => r.a }, { key: 'b', label: 'Within-day blocked', align: 'right', render: (r) => r.b }]} />
        </Card>
      </div>
      <Card title="Secondary: within-day blocked split" subtitle={`${sb.split}. Easier by construction; never used to pick a model.`}>
        <ModelTable pool={sb.pooled} />
      </Card>
    </div>
  )
}

function Method({ m }: { m: DaptMeta }) {
  const steps = [
    ['Ingest', 'CICFlowMeter-style flow CSVs: fix headerless files, unify labels, parse timestamps, drop flows duplicated between capture points.'],
    ['Entity-centric state', `Each monitored (private-address) host gets one feature vector per ${m.bucket_s}s bucket: inbound and outbound flows, distinct peers and ports, packets, bytes, TCP flag counts and port-class shares. Signed log1p on heavy tails; scalers fit on training rows only. IPs, raw ports and timestamps are never inputs.`],
    ['Dynamics', `A residual GRU learns P(next state | last ${m.L} minutes), then is rolled forward ${m.K} steps on its own predictions.`],
    ['Forecast', `A gradient-boosted head reads the current state, the simulated future and the latent state to output P(attack in the next ${m.K} minutes). A separate model maps a forecast onset to a kill-chain stage.`],
    ['Explain', 'Exact TreeSHAP attributions per prediction, plus simulated-vs-observed trajectories.'],
    ['Respond', 'Alerts become incidents with MITRE D3FEND countermeasures and containment templates on the Incident response page.'],
  ]
  return (
    <div className="space-y-5">
      <Card title="Pipeline" subtitle="The same code trains, benchmarks, serves this console and the CLI">
        <ol className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {steps.map(([t, b], k) => (
            <li key={t} className="rounded-lg border border-line p-4">
              <div className="flex items-center gap-2"><span className="flex size-6 items-center justify-center rounded-full bg-accent-wash text-[12px] font-semibold text-accent-ink">{k + 1}</span><span className="font-semibold">{t}</span></div>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{b}</p>
            </li>
          ))}
        </ol>
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Stage mapping (DAPT2020 → ATT&CK tactic)">
          <Table rows={[['Reconnaissance', 'Reconnaissance'], ['Establish Foothold', 'Initial Access'], ['Lateral Movement', 'Lateral Movement'], ['Data Exfiltration', 'Exfiltration']]} rowKey={(r) => r[0]}
            columns={[{ key: 'a', label: 'DAPT label', render: (r) => r[0] }, { key: 'b', label: 'ATT&CK tactic', render: (r) => <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: STAGE_COLOR[r[1]] }} />{r[1]}</span> }]} />
          <p className="mt-2 text-[12px] text-muted">DAPT has no Command &amp; Control label.</p>
        </Card>
        <Note title="Evaluation discipline">
          Leave-one-day-out is primary; the blocked split is secondary. Alarm budgets are set on held-out benign minutes. Every experiment has its hypothesis and prediction in <code>results/registry.csv</code>, written before the run. Leakage tests: <code>make test</code>.
        </Note>
      </div>
    </div>
  )
}
