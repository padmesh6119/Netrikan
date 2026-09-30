// Categorical slots in fixed order (validated palette); an entity keeps its slot regardless of what is filtered.
export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)']

export const STAGE_COLOR: Record<string, string> = {
  Reconnaissance: SERIES[0],
  'Initial Access': SERIES[1],
  'Lateral Movement': SERIES[2],
  Exfiltration: SERIES[3],
}

export const TECH_COLOR: Record<string, string> = {
  T1595: SERIES[0],
  T1190: SERIES[1],
  T1078: SERIES[2],
  T1110: SERIES[3],
  T1048: SERIES[4],
}

/** Nine-step sequential ramp (single hue), index 0 = near zero. */
export const seq = (v: number) => `var(--seq-${Math.max(0, Math.min(8, Math.round(v * 8)))})`
