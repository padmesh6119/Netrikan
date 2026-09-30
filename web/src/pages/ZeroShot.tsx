import { ScoreBar, Table } from '../components/Table'
import { Card, ErrorState, Findings, Loading, Note, PageHeader, Stat } from '../components/ui'
import { useApi, type Detector, type ZeroShot } from '../lib/api'
import { int, num } from '../lib/format'

const SHORT: Record<Detector, string> = { iforest: 'IsolationForest', surprise: 'World-model surprise', gbdt: 'Supervised GBDT', volume: 'Volume heuristic', fanout: 'Fan-out heuristic' }

export default function ZeroShotPage() {
  const q = useApi<ZeroShot>('/api/zeroshot')
  if (q.error) return <ErrorState error={q.error} />
  if (!q.data) return <Loading />
  const d = q.data
  const dets = (Object.keys(d.detectors) as Detector[]).filter((k) => d.rows.every((r) => r.auc[k] != null))
  const mean = (k: Detector) => d.rows.reduce((a, r) => a + (r.auc[k] ?? 0), 0) / d.rows.length
  const below = d.rows.filter((r) => (r.auc.gbdt ?? 1) < 0.5).length
  const bestMean = Math.max(...dets.map(mean))
  return (
    <>
      <PageHeader title="Zero-shot transfer across labs">
        Four corpora from different labs, capture tools and years. For each, detectors are trained on the other three and scored on the held-out one, so every attack family in the test set is unseen. A learned detector only counts if it beats the two heuristics.
      </PageHeader>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        {dets.map((k) => (
          <Stat key={k} label={SHORT[k]} value={num(mean(k))} hint={mean(k) === bestMean ? 'Best mean ROC-AUC' : 'Mean ROC-AUC, unseen families'} tone={mean(k) === bestMean ? 'good' : mean(k) < 0.5 ? 'critical' : undefined} />
        ))}
      </div>
      <Card title="What this shows, and what it does not">
        <Findings items={[
          { title: 'Learning attacks from other labs does not transfer.', body: <>The supervised detector averages {num(mean('gbdt'))} and is below chance on {below} of {d.rows.length} unseen families. Attack signatures from one lab invert in another.</> },
          { title: 'Benign-only anomaly detection transfers best.', body: <>IsolationForest averages {num(mean('iforest'))}; the plain flow-volume heuristic {num(mean('volume'))}. Scans, DDoS and brute force are visibly abnormal without any attack training.</> },
          { title: `World-model surprise averages ${num(mean('surprise'))}.`, body: <>{mean('surprise') <= mean('iforest') + 0.03 ? 'It does not beat IsolationForest here.' : 'It beats IsolationForest.'} Stealthy families (botnet C&amp;C, Heartbleed, DAPT lateral movement) stay near chance for every detector.</> },
          { title: 'Caveats.', body: <>Positive minutes are tens per family, host-minutes within a host are not independent, and negatives are the held-out corpus's own benign minutes. Small differences are noise; the ordering supervised &lt; heuristics &lt; IsolationForest is the robust part.</> },
          { title: 'Design consequence.', body: <>For attacks the model has never seen, use anomaly-style scoring, not a classifier trained on other labs' attacks.</> },
        ]} />
      </Card>
      <Card title="ROC-AUC per unseen attack family" subtitle="Tick = 0.5 (chance). Highlighted = best detector for that family.">
        <Table rows={d.rows} rowKey={(r) => r.corpus + r.family} columns={[
          { key: 'c', label: 'Held-out corpus', value: (r) => r.corpus },
          { key: 'f', label: 'Unseen family', value: (r) => r.family },
          { key: 'n', label: 'Positive min', align: 'right', value: (r) => r.positives, render: (r) => int(r.positives) },
          ...dets.map((k) => ({
            key: k, label: SHORT[k], align: 'right' as const, value: (r: ZeroShot['rows'][number]) => r.auc[k],
            render: (r: ZeroShot['rows'][number]) => <ScoreBar value={r.auc[k]} reference={0.5} best={r.auc[k] === Math.max(...dets.map((x) => r.auc[x] ?? 0))} format={(v) => v.toFixed(2)} />,
          })),
          { key: 'ci', label: 'GBDT 90% CI', align: 'right', render: (r) => (r.gbdt_ci90 ? `${num(r.gbdt_ci90[0])}-${num(r.gbdt_ci90[1])}` : '-') },
        ]} />
      </Card>
      <Card title="Protocol">
        <p className="text-[13px] text-ink-2">{d.protocol}</p>
        <div className="mt-4">
          <Table dense rows={Object.entries(d.corpora)} rowKey={([k]) => k} columns={[
            { key: 'l', label: 'Corpus', render: ([, v]) => v.label },
            { key: 'h', label: 'Host-minutes', align: 'right', render: ([, v]) => int(v.host_minutes) },
            { key: 'a', label: 'Attack host-minutes', align: 'right', render: ([, v]) => int(v.attack_host_minutes) },
          ]} />
        </div>
        {d.note && <div className="mt-4"><Note>{d.note}</Note></div>}
      </Card>
    </>
  )
}
