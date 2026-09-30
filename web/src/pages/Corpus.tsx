import { useParams } from 'react-router-dom'
import { Heatmap, RampLegend } from '../components/charts/Heatmap'
import { TimeChart, toBands } from '../components/charts/TimeChart'
import { ScoreBar, Table } from '../components/Table'
import { Busy, Card, ErrorState, Field, Legend, Loading, Note, PageHeader, Segmented, Select, Stat, StatRow, Tabs, Toolbar } from '../components/ui'
import { useApi, type CorpusData, type CorpusHost, type CorpusMeta, type Detector } from '../lib/api'
import { SERIES } from '../lib/colors'
import { int, num, pct } from '../lib/format'
import { useUrlState } from '../lib/url'

type Tab = 'timeline' | 'compare' | 'heatmap'
const DET_SHORT: Record<Detector, string> = { iforest: 'IsolationForest', surprise: 'World-model surprise', gbdt: 'Supervised GBDT', volume: 'Volume heuristic', fanout: 'Fan-out heuristic' }

export default function CorpusPage() {
  const { name = 'cic17' } = useParams()
  const { get, set } = useUrlState()
  const meta = useApi<CorpusMeta>(`/api/corpus/${name}/meta`)
  const m = meta.data
  const slice = get('slice', m ? Object.keys(m.slices)[0] : 'both')
  const det = get('det', 'iforest') as Detector
  const budget = Number(get('budget', '2'))
  const tab = get('tab', 'timeline') as Tab
  const c = useApi<CorpusData>(`/api/corpus/${name}`, { slice, det, budget }, !!m)

  if (meta.error) return <ErrorState error={meta.error} />
  if (!m) return <Loading />
  const d = c.data
  const famColor = (k: number) => SERIES[k % SERIES.length]
  return (
    <>
      <Busy on={c.isFetching} />
      <PageHeader title={m.label}>
        Scored by detectors that never saw this corpus (trained on {m.trained_on.join(', ')}). Nothing about this lab's traffic or attacks was used for training or threshold selection. Labels are used only to draw attack minutes and to score.
      </PageHeader>
      <Toolbar>
        {Object.keys(m.slices).length > 1 && (
          <Field label="Slice"><Select value={slice} onChange={(v) => set({ slice: v, host: null })} options={Object.entries(m.slices).map(([k, v]) => ({ value: k, label: v }))} /></Field>
        )}
        <Field label="Detector"><Select value={det} onChange={(v) => set({ det: v })} options={(Object.keys(m.detectors) as Detector[]).map((k) => ({ value: k, label: m.detectors[k] }))} /></Field>
        <Field label="Alert budget" hint="Label-free: the top-scoring host-minutes, this many per host-hour">
          <Segmented value={budget} onChange={(v) => set({ budget: v })} items={m.budgets.map((b) => ({ id: b, label: `${b}/h` }))} />
        </Field>
      </Toolbar>
      {c.error && <ErrorState error={c.error} />}
      {!d && !c.error && <Loading label="Loading flows, building host-minutes and scoring with the held-out detectors" />}
      {d && (
        <>
          <StatRow>
            <Stat label="Flows" value={int(d.n_flows)} />
            <Stat label="Hosts / host-minutes" value={`${int(d.n_hosts)} / ${int(d.host_minutes)}`} />
            <Stat label="Attack host-minutes" value={int(d.attack_minutes)} />
            <Stat label="Attack minutes caught" value={pct(d.caught)} hint={`${int(d.alerts)} alerts`} tone={d.caught > 0.5 ? 'good' : undefined} />
            <Stat label="False alerts / benign host-hour" value={num(d.false_alerts_per_host_hour)} />
          </StatRow>
          <Tabs value={tab} onChange={(v) => set({ tab: v })} items={[{ id: 'timeline', label: 'Host timeline' }, { id: 'compare', label: 'Detector comparison' }, { id: 'heatmap', label: 'Network heatmap' }]} />
          {tab === 'timeline' && <HostTimeline name={name} slice={slice} det={det} budget={budget} d={d} famColor={famColor} />}
          {tab === 'compare' && (
            <Card title="ROC-AUC on this slice, per attack family" subtitle="0.5 = chance (tick). Computed live. Attack minutes are tens and minutes within a host are not independent: differences of a few hundredths are noise.">
              <Table rows={d.comparison} rowKey={(r) => r.family} columns={[
                { key: 'f', label: 'Family', render: (r) => r.family },
                { key: 'n', label: 'Positive min', align: 'right', value: (r) => r.positives },
                { key: 'b', label: 'Base rate', align: 'right', render: (r) => num(r.base_rate, 4) },
                ...(Object.keys(m.detectors) as Detector[]).map((k) => ({
                  key: k, label: DET_SHORT[k], align: 'right' as const, value: (r: CorpusData['comparison'][number]) => r.auc[k],
                  render: (r: CorpusData['comparison'][number]) => <ScoreBar value={r.auc[k]} reference={0.5} best={r.auc[k] === Math.max(...Object.values(r.auc))} format={(v) => v.toFixed(2)} />,
                })),
              ]} />
            </Card>
          )}
          {tab === 'heatmap' && (
            <Card title="The 14 highest-scoring hosts" subtitle={`${m.detectors[det]}, as a percentile within this slice`} actions={<div className="flex items-center gap-4"><RampLegend label="Score percentile" lo="0" hi="1" /><Legend items={[{ label: 'True attack minute', color: 'var(--critical)' }]} /></div>}>
              <Heatmap rows={d.heatmap.hosts} host={d.heatmap.host} t={d.heatmap.t} v={d.heatmap.v} marker={d.heatmap.attack} valueLabel="Percentile" onRowClick={(h) => set({ tab: 'timeline', host: h })} />
            </Card>
          )}
        </>
      )}
    </>
  )
}

function HostTimeline({ name, slice, det, budget, d, famColor }: { name: string; slice: string; det: Detector; budget: number; d: CorpusData; famColor: (k: number) => string }) {
  const { get, set } = useUrlState()
  const host = get('host', d.hosts[0]?.host)
  const h = useApi<CorpusHost>(`/api/corpus/${name}/host`, { host, slice, det, budget }, !!host)
  return (
    <Card
      title={<span className="flex items-center gap-3">Score over time<Select value={host} onChange={(v) => set({ host: v })} className="h-8 font-mono text-[13px]" options={d.hosts.map((x) => ({ value: x.host, label: x.attack_minutes ? `${x.host} · ${x.attack_minutes} attack min` : x.host }))} /></span>}
      subtitle="Detector score as a percentile within this slice, so detectors are comparable. Shaded: true attack minutes by family."
      actions={<Legend items={[{ label: 'Score percentile', color: 'var(--s1)', shape: 'line' }, { label: 'Alert', color: 'var(--critical)', shape: 'dot' }, ...d.families.map((f, k) => ({ label: f, color: famColor(k) }))]} />}>
      {h.error && <ErrorState error={h.error} />}
      {h.data ? (
        <TimeChart height={320} yMax={1} yFormat={(v) => v.toFixed(2)}
          series={[{ id: 'p', label: 'Percentile', color: 'var(--s1)', t: h.data.t, v: h.data.pct }]}
          bands={toBands(h.data.t, h.data.family.map((f) => (f < 0 ? -1 : f + 1)), (c) => famColor((c as number) - 1), (c) => d.families[(c as number) - 1])}
          points={h.data.t.flatMap((t, k) => (h.data!.alarm[k] ? [{ t, v: h.data!.pct[k], color: 'var(--critical)' }] : []))}
          threshold={{ v: h.data.threshold, label: 'alert cut' }} />
      ) : <Loading />}
      <div className="mt-3"><Note>Gaps in the line are minutes with no traffic from this host.</Note></div>
    </Card>
  )
}
