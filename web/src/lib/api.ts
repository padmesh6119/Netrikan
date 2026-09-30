import { keepPreviousData, useQuery } from '@tanstack/react-query'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    let msg = r.statusText
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, msg)
  }
  return r.json()
}

export function qs(params: Record<string, string | number | boolean | undefined | null>) {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  return p.toString()
}

/** GET a JSON endpoint; `enabled` false while a dependency (host, row) is not chosen yet. */
export function useApi<T>(path: string, params: Record<string, string | number | boolean | undefined | null> = {}, enabled = true) {
  const url = `${path}${Object.keys(params).length ? `?${qs(params)}` : ''}`
  return useQuery<T, ApiError>({
    queryKey: [url],
    queryFn: () => request<T>(url),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: (n, e) => n < 1 && !(e instanceof ApiError && e.status < 500),
  })
}

export async function uploadFlows(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return request<{ source: string; name: string; n_flows: number }>('/api/dapt/upload', { method: 'POST', body: fd })
}

// ---------------------------------------------------------------- DAPT2020
export type DaptModel = 'lag' | 'world'
export interface DaptMeta {
  days: string[]
  onsets_by_day: Record<string, number>
  K: number
  L: number
  bucket_s: number
  primary: DaptModel
  stages: string[]
  thresholds: Record<DaptModel, number>
  uploads: { source: string; name: string; n_flows: number }[]
}
export interface DaptScores {
  source: string
  model_name: string
  n_flows: number
  labelled: boolean
  hosts: string[]
  bucket_s: number
  K: number
  rows: { host: number[]; t: number[]; lag: number[]; world: number[]; attack: number[]; stage: number[] }
}
export interface DaptSummary {
  n_flows: number
  n_hosts: number
  host_minutes: number
  alert_minutes: number
  labelled: boolean
  false_alarms_per_host_hour?: number
  onsets?: number
  onsets_warned?: number
  median_lead_min?: number | null
  onset_list?: { host: string; t: number; stage: string; lead_min: number }[]
}
export interface Driver {
  feature: string
  value: number
}
export interface DaptExplain {
  i: number
  host: string
  t: number
  p: number
  outcome: 'attack_now' | 'attack_next' | 'quiet' | 'past_end' | null
  true_stage: string | null
  stage_probs: { stage: string; p: number }[]
  stage_accuracy: number
  stage_majority: number
  drivers: Driver[]
  rollout: { feature: string; forecast: number[]; observed: (number | null)[] }[]
  flows: { columns: string[]; rows: (string | number | boolean | null)[][] }
}

// ---------------------------------------------------------------- ZeekData24
export interface Technique {
  id: string
  name: string
  tactic: string
}
export type Rung = 'sched' | 'world' | 'renewal'
export interface Z24Meta {
  weeks: { week: string; kind: 'attack' | 'benign' }[]
  K: number
  techniques: Technique[]
  rungs: Record<Rung, string>
  budgets: number[]
}
export interface Z24Week {
  week: string
  is_attack: boolean
  n_flows: number
  traffic_minutes: number
  host_minutes: number
  flagged_any: number
  benign_false_alarms_per_host_hour: number | null
  attacker_sequences: number
  techniques: (Technique & { flagged: number; true?: number; recall?: number; precision?: number })[]
  flagged_hosts: { host: string; minutes: number }[]
  attackers: { host: string; minutes: number; days: string[] }[]
}
export interface Z24RecogHost {
  ticks: { id: string; recognized: number[]; truth: number[] }[]
  top: { i: number; t: number; p: number[] }[]
}
export interface Z24RecogExplain {
  i: number
  t: number
  probs: { id: string; name: string; p: number }[]
  explained: string
  drivers: Driver[]
}
export interface Z24Forecast {
  days: string[]
  day: string
  K: number
  threshold: number
  series: { t: number[]; i: number[]; model: number[]; baseline: number[] }
  bursts: number[]
  alerts: { t: number[]; p: number[] }
  stats: { onsets: number; warned: number | null; false_alarms_per_attacker_hour: number; median_lead_min: number | null }
  top: { i: number; t: number; p: number }[]
}
export interface Z24Structure {
  is_attack: boolean
  uniform_std: number
  minute_std: number
  lags: { techs: string[]; matrix: (number | null)[][] }
  attackers: Record<string, string | number>[]
  minute_hist: { id: string; counts: number[] }[]
  gap_hist: { id: string; counts: number[] }[]
}

// ---------------------------------------------------------------- unseen corpora
export type Detector = 'iforest' | 'surprise' | 'gbdt' | 'volume' | 'fanout'
export interface CorpusMeta {
  name: string
  label: string
  trained_on: string[]
  slices: Record<string, string>
  detectors: Record<Detector, string>
  budgets: number[]
}
export interface CorpusData {
  n_flows: number
  n_hosts: number
  host_minutes: number
  attack_minutes: number
  alerts: number
  caught: number
  false_alerts_per_host_hour: number
  threshold: number
  families: string[]
  hosts: { host: string; attack_minutes: number }[]
  comparison: { family: string; positives: number; base_rate: number; auc: Record<Detector, number> }[]
  heatmap: { hosts: string[]; host: number[]; t: number[]; v: number[]; attack: number[] }
}
export interface CorpusHost {
  threshold: number
  t: number[]
  pct: number[]
  alarm: number[]
  family: number[]
}
export interface ZeroShot {
  protocol: string
  note: string | null
  corpora: Record<string, { label: string; host_minutes: number; attack_host_minutes: number }>
  detectors: Record<Detector, string>
  rows: { corpus: string; family: string; positives: number; base_rate: number; auc: Partial<Record<Detector, number>>; gbdt_ci90: [number, number] | null }[]
}

// ---------------------------------------------------------------- response (ATT&CK -> D3FEND)
export type Priority = 'P1' | 'P2' | 'P3'
export interface Incident {
  id: string
  host: string
  start: number
  end: number
  row: number
  peak: number
  ratio: number
  minutes: number
  priority: Priority
  score: number
  tactic: string
  playbook_key: string
  tactic_p: number | null
  technique: string | null
  recent?: string[]
  outcome?: 'attack' | 'no_attack'
  lead_min?: number | null
  true_tactic?: string | null
}
export interface IncidentList {
  dataset: 'dapt' | 'z24'
  source: string
  labelled: boolean
  note: string
  threshold: number | null
  incidents: Incident[]
}
export interface Countermeasure {
  d3fend: string
  tactic: 'Detect' | 'Isolate' | 'Deceive' | 'Evict' | 'Harden' | 'Restore' | 'Model'
  action: string
  why: string
}
export interface IncidentDetail {
  incident: Incident
  drivers: Driver[]
  targets: {
    flows: number
    out_peers: { value: string; flows: number }[]
    out_ports: { value: string; flows: number }[]
    in_peers: { value: string; flows: number }[]
    in_ports: { value: string; flows: number }[]
    internal_peers: { value: string; flows: number }[]
    external_by_bytes: { value: string; bytes: number }[]
    admin_ports: number[]
    n_out_peers: number
    n_out_ports: number
    bytes_out: number
    bytes_in: number
  }
  confidence: { level: 'low' | 'high'; text: string }
  playbook: { group: string; confidence: 'low' | 'high'; items: Countermeasure[] }[]
  rules: string[]
  series: { t: number[]; p: number[]; attack: number[] }
  attack_ref: { tactic: string; technique: string | null; technique_name: string | null; technique_tactic: string | null }
}
