"""
Measure stage persistence from DAPT 2020, and check it against what
forecast.MEASURED_PERSISTENCE currently declares.

`forecast.MEASURED_PERSISTENCE` is the diagonal of the transition matrix -- the
probability a stage is followed by itself. The values were taken from DAPT, but
the mapping was made when Netrikan had no separate Reconnaissance stage: DAPT's
Reconnaissance figure was assigned to INITIAL_ACCESS, and DAPT's Establish
Foothold figure was assigned to C2. Now that RECON exists, that mapping needs
re-deriving from the data rather than by judgement.

Two orderings are reported, because they answer different questions:

  per capture   consecutive flows inside one capture file, network-wide. This is
                what dapt.transition_counts does and what the current numbers
                came from. It mixes unrelated hosts: flow t and t+1 are often
                different machines, so "persistence" here partly measures how
                long the capture stays in one phase overall.

  per host      consecutive flows of the SAME Src IP. This is the quantity the
                forecast actually needs -- P(this host is still in this stage on
                its next flow) -- and it is the honest input to a per-host
                kill-chain projection.

Nothing is written to forecast.py. This prints a recommendation; changing the
matrix changes every published forecast number, so it is a deliberate edit.

Usage:
    python3 bench/dapt_persistence.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import dapt                     # noqa: E402
import forecast as fc           # noqa: E402
from attck_map import (BENIGN, INITIAL_ACCESS, DOS, LATERAL, C2, EXFIL, RECON,
                       short)   # noqa: E402

# DAPT's five phases -> Netrikan's chain stages.
# DAPT has no DoS phase, so DOS gets no measurement from this corpus.
DAPT_TO_CHAIN = {
    'Benign': BENIGN,
    'Reconnaissance': RECON,
    'Establish Foothold': INITIAL_ACCESS,   # gaining a foothold IS initial access
    'Lateral Movement': LATERAL,
    'Data Exfiltration': EXFIL,
}

MIN_OBSERVATIONS = 200          # below this, a diagonal estimate is noise


def diagonals(df, within):
    """P(same stage on the next flow | current stage), grouped by `within`."""
    n = len(dapt.DAPT_STAGES)
    same = np.zeros(n, dtype=np.int64)
    total = np.zeros(n, dtype=np.int64)
    for _, grp in df.groupby(within, sort=False):
        g = grp.sort_values('ts', kind='mergesort')
        s = g['stage_id'].to_numpy()
        if len(s) < 2:
            continue
        np.add.at(total, s[:-1], 1)
        np.add.at(same, s[:-1], (s[:-1] == s[1:]).astype(np.int64))
    return same, total


def main():
    df = dapt.load_all()
    df = df.dropna(subset=['ts'])
    print(f"DAPT 2020: {len(df):,} flows, {df['capture'].nunique()} captures, "
          f"{df['Src IP'].nunique():,} source hosts\n")

    results = {}
    for label, within in (('per_capture', 'capture'), ('per_host', 'Src IP')):
        same, total = diagonals(df, within)
        print(f"--- {label} ---")
        rows = {}
        for sid, name in enumerate(dapt.DAPT_STAGES):
            if total[sid] == 0:
                print(f"  {name:<22} no transitions observed")
                continue
            p = float(same[sid] / total[sid])
            flag = "" if total[sid] >= MIN_OBSERVATIONS else "  <-- too few, noise"
            print(f"  {name:<22} n={total[sid]:>7,}  persistence={p:.4f}{flag}")
            rows[name] = {'n': int(total[sid]), 'persistence': round(p, 4),
                          'reliable': bool(total[sid] >= MIN_OBSERVATIONS)}
        results[label] = rows
        print()

    # ---- compare against what forecast.py declares ----
    print("--- forecast.MEASURED_PERSISTENCE vs per-host measurement ---")
    per_host = results['per_host']
    comparison, recommend = {}, {}
    for dapt_name, chain_id in DAPT_TO_CHAIN.items():
        declared = fc.MEASURED_PERSISTENCE[chain_id]
        row = per_host.get(dapt_name)
        if row is None:
            continue
        measured, reliable = row['persistence'], row['reliable']
        delta = measured - declared
        mark = "OK" if abs(delta) < 0.02 else "DIFFERS"
        if not reliable:
            mark = "unreliable"
        print(f"  {short(chain_id):<18} declared={declared:.3f}  "
              f"measured={measured:.4f}  delta={delta:+.4f}  [{mark}]  "
              f"(from DAPT '{dapt_name}', n={row['n']:,})")
        comparison[short(chain_id)] = {
            'declared': declared, 'measured_per_host': measured,
            'delta': round(delta, 4), 'dapt_phase': dapt_name,
            'n': row['n'], 'reliable': reliable, 'status': mark,
        }
        if reliable:
            recommend[short(chain_id)] = round(measured, 3)

    print(f"\n  {short(DOS):<18} not present in DAPT — "
          f"declared {fc.MEASURED_PERSISTENCE[DOS]:.3f} is an assumption, "
          f"not a measurement")

    print("\nRecommended diagonal (reliable per-host estimates only):")
    for k, v in recommend.items():
        print(f"  {k:<18} {v}")
    print("\nNot applied automatically: editing MEASURED_PERSISTENCE changes every "
          "published forecast number, so make it a deliberate commit.")

    out = os.path.join(ROOT, 'models', 'dapt_persistence.json')
    with open(out, 'w') as f:
        json.dump({
            'n_flows': int(len(df)),
            'n_captures': int(df['capture'].nunique()),
            'n_hosts': int(df['Src IP'].nunique()),
            'min_observations_for_reliable': MIN_OBSERVATIONS,
            'dapt_phase_to_chain_stage': {k: short(v)
                                          for k, v in DAPT_TO_CHAIN.items()},
            'measured': results,
            'comparison_with_forecast_py': comparison,
            'recommended_diagonal': recommend,
            'note': ('per_host is the quantity the forecast needs: P(this host is '
                     'still in this stage on its next flow). per_capture mixes '
                     'unrelated hosts and is what the current values came from. '
                     'DoS is absent from DAPT entirely, so its declared 0.900 is '
                     'an assumption.'),
        }, f, indent=2)
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
