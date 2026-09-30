import { ArrowRight } from '@phosphor-icons/react'
import { motion } from 'motion/react'
import { Link } from 'react-router-dom'
import { PageHeader } from '../components/ui'
import { useApi, type CorpusData, type IncidentList, type ZeroShot } from '../lib/api'
import { num, pct } from '../lib/format'

interface DM { pooled: Record<string, Record<string, number>>; stage: { accuracy: number; majority_baseline_accuracy: number } }
interface ZM { forecast: { macro: Record<string, Record<string, number>> } }

function Row({ to, title, body, metric, metricLabel }: { to: string; title: string; body: string; metric?: string; metricLabel?: string }) {
  return (
    <Link to={to} className="group grid items-center gap-x-8 gap-y-2 px-5 py-5 transition-colors duration-200 hover:bg-surface-hover sm:grid-cols-[minmax(0,1fr)_12rem_1.5rem]">
      <div className="min-w-0">
        <div className="text-[15px] font-semibold tracking-[-0.01em]">{title}</div>
        <p className="mt-1 max-w-[70ch] text-[13.5px] leading-relaxed text-ink-2">{body}</p>
      </div>
      <div className="sm:text-right">
        {metric ? (
          <>
            <div className="text-[20px] font-semibold tracking-tight tnum">{metric}</div>
            <div className="text-[12px] text-muted">{metricLabel}</div>
          </>
        ) : metricLabel ? <div className="text-[12px] text-muted">{metricLabel}</div> : null}
      </div>
      <ArrowRight className="hidden size-4 text-muted transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-ink sm:block" />
    </Link>
  )
}

export default function DashboardHome() {
  const dm = useApi<DM>('/api/dapt/metrics')
  const zm = useApi<ZM>('/api/z24/metrics')
  const zs = useApi<ZeroShot>('/api/zeroshot')
  const inc = useApi<IncidentList>('/api/response/incidents', { dataset: 'dapt', source: '2019-07-17', model: 'lag', thr: 0.48 })
  const lag = dm.data?.pooled.lag, prev = dm.data?.pooled.lr.prevalence
  const sched = zm.data?.forecast.macro.sched
  const iso = zs.data ? zs.data.rows.reduce((a, r) => a + (r.auc.iforest ?? 0), 0) / zs.data.rows.length : undefined
  const p1 = inc.data?.incidents.filter((i) => i.priority === 'P1').length
  const cic = useApi<CorpusData>('/api/corpus/cic17', { slice: 'fri', det: 'iforest', budget: 2 })
  const ctu = useApi<CorpusData>('/api/corpus/ctu13', { slice: 'both', det: 'iforest', budget: 2 })
  return (
    <>
      <PageHeader title="Dashboard">
        Every view runs the trained models on held-out data. Start with a forecast, then follow an alert into the response queue.
      </PageHeader>

      <motion.section initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.08, ease: [0.16, 1, 0.3, 1] }} aria-labelledby="h-forecast">
        <h2 id="h-forecast" className="mb-2 text-[13px] font-medium text-muted">Forecast</h2>
        <div className="divide-y divide-line overflow-hidden rounded-xl border border-line bg-surface">
          <Row to="/dashboard/dapt" title="DAPT2020 host forecasting" body="Per-host risk that an attack starts in the next 5 minutes, with stage, TreeSHAP drivers and the world model's simulated future."
            metric={lag && prev ? `${num(lag.pr_auc / prev, 1)}x` : undefined} metricLabel="PR-AUC over base rate" />
          <Row to="/dashboard/zeek" title="ZeekData24 attacker campaigns" body="Recognise the ATT&CK technique in each host-minute, then forecast each attacker's next burst from its own history."
            metric={sched ? `${num(sched.pr_auc / sched.prevalence, 1)}x` : undefined} metricLabel="Next-burst PR-AUC over base rate" />
          <Row to="/dashboard/labs/cic17" title="CIC-IDS2017" body="Port scan, DDoS and Heartbleed. Per-host anomaly score each minute, with attack minutes and alerts on a timeline."
            metric={cic.data ? pct(cic.data.caught) : undefined} metricLabel="Attack minutes caught, Friday, 2 alerts/h" />
          <Row to="/dashboard/labs/ctu13" title="CTU-13 scenario 4" body="Botnet spam, ICMP and C&C traffic from a 2011 university network, scored the same way."
            metric={ctu.data ? pct(ctu.data.caught) : undefined} metricLabel="Attack minutes caught, 2 alerts/h" />
        </div>
      </motion.section>

      <motion.section initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.16, ease: [0.16, 1, 0.3, 1] }} aria-labelledby="h-respond">
        <h2 id="h-respond" className="mb-2 text-[13px] font-medium text-muted">Respond</h2>
        <div className="overflow-hidden rounded-xl border border-line bg-surface">
          <Row to="/dashboard/response" title="Incident response" body="Alerts become prioritised incidents with MITRE D3FEND countermeasures and containment rules built from the host's real peers and ports."
            metric={p1 != null ? String(p1) : undefined} metricLabel="P1 incidents on Wed 17 Jul" />
        </div>
      </motion.section>

      <motion.section initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.24, ease: [0.16, 1, 0.3, 1] }} aria-labelledby="h-evidence">
        <h2 id="h-evidence" className="mb-2 text-[13px] font-medium text-muted">Evidence</h2>
        <div className="overflow-hidden rounded-xl border border-line bg-surface">
          <Row to="/dashboard/zero-shot" title="Zero-shot transfer" body="Leave-one-corpus-out across four labs: what generalises to attacks nobody trained on, and what does not."
            metric={iso != null ? num(iso) : undefined} metricLabel="Mean ROC-AUC, IsolationForest" />
        </div>
      </motion.section>

      {dm.data && (
        <p className="max-w-[80ch] text-[12.5px] leading-relaxed text-muted">
          Scope: DAPT2020 has 9 monitored hosts; ZeekData24 replays one scripted campaign. Stage forecasting is at chance across held-out days
          ({pct(dm.data.stage.accuracy)} accuracy vs {pct(dm.data.stage.majority_baseline_accuracy)} for the majority class), so stage playbooks are labelled as hypotheses.
        </p>
      )}
    </>
  )
}
