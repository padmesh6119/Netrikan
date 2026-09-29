"""
Lead time against Suricata, an external clock the model cannot influence.

infer.analyze() reports lead_seconds as
(detect_idx - forecast_idx) * flow_interval. Both indices come from the same
model's output, so that number compares Netrikan's forecast threshold to
Netrikan's own detection threshold. It is arithmetic, not a measurement.

This script replaces it with a real baseline: run Suricata offline over the same
PCAP, take its first alert, and compare against the first time Netrikan could
have warned.

The conservative accounting matters and is the headline here. Window i spans
flows i..i+WINDOW, so a forecast at window i is only available once flow
i+WINDOW has been observed. Reporting forecast_idx * interval would credit
Netrikan with a warning it could not yet have made, overstating the lead by
WINDOW * interval. Both numbers are reported; `lead_seconds_conservative` is the
defensible one.

A negative lead means Suricata fired first. That is a real outcome and it is
recorded as-is.

Usage:
    python3 bench/suricata_comparison.py demo_attack_lab/attack_small.pcap
    python3 bench/suricata_comparison.py <pcap> --suricata-log /tmp/sout
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import forecast as fc          # noqa: E402
import infer                   # noqa: E402
from pcap_ingest import load_pcap   # noqa: E402


def run_suricata(pcap, logdir, config=None):
    if not shutil.which('suricata'):
        return None, "suricata not on PATH"
    cmd = ['suricata', '-r', pcap, '-l', logdir, '-k', 'none']
    if config:
        cmd += ['-c', config]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, "suricata timed out"
    eve = os.path.join(logdir, 'eve.json')
    if not os.path.exists(eve):
        return None, (f"suricata produced no eve.json (exit {p.returncode}). "
                      f"stderr: {p.stderr.strip()[:300]}")
    return eve, None


def parse_eve(eve_path):
    """First alert plus the total count. Timestamps are ISO8601 with offset."""
    alerts = []
    with open(eve_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get('event_type') == 'alert':
                alerts.append(e)
    if not alerts:
        return None, 0
    alerts.sort(key=lambda e: e['timestamp'])
    return alerts[0], len(alerts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pcap')
    ap.add_argument('--suricata-config', default=None)
    ap.add_argument('--suricata-log', default=None,
                    help='reuse an existing Suricata log dir instead of running it')
    ap.add_argument('--horizon', type=float, default=fc.HORIZON_SECONDS)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models',
                                                 'suricata_comparison.json'))
    args = ap.parse_args()

    if not os.path.exists(args.pcap):
        raise SystemExit(f"no such PCAP: {args.pcap}")

    # ---- Suricata ----
    if args.suricata_log:
        eve, err = os.path.join(args.suricata_log, 'eve.json'), None
        if not os.path.exists(eve):
            raise SystemExit(f"no eve.json in {args.suricata_log}")
        tmp = None
    else:
        tmp = tempfile.mkdtemp(prefix='netrikan_suricata_')
        print(f"running suricata -> {tmp}", flush=True)
        eve, err = run_suricata(args.pcap, tmp, args.suricata_config)
    if err:
        raise SystemExit(f"suricata unavailable: {err}")

    first_alert, n_alerts = parse_eve(eve)
    if first_alert is None:
        print("suricata fired 0 alerts on this PCAP", flush=True)
    else:
        print(f"suricata: {n_alerts} alerts, first at {first_alert['timestamp']} "
              f"({first_alert['alert']['signature']})", flush=True)

    # ---- Netrikan ----
    print("running netrikan", flush=True)
    df = load_pcap(args.pcap)
    if df.empty:
        raise SystemExit("pcap_ingest returned no flows")
    r = infer.analyze(df, args.horizon)
    if r is None:
        raise SystemExit(f"need more than {infer.WINDOW} flows; got {len(df)}")

    ts = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
    t0 = ts.min()
    interval = r['flow_interval']
    fi = r['forecast_idx']

    optimistic = fi * interval if fi is not None else None
    # a forecast at window fi needs flows fi..fi+WINDOW observed first
    conservative = (fi + infer.WINDOW) * interval if fi is not None else None

    suricata_offset = None
    if first_alert is not None:
        sts = pd.to_datetime(first_alert['timestamp'], errors='coerce',
                             format='mixed')
        if pd.notna(sts) and pd.notna(t0):
            if sts.tzinfo is not None and t0.tzinfo is None:
                sts = sts.tz_localize(None)
            suricata_offset = float((sts - t0).total_seconds())

    lead_cons = lead_opt = None
    if suricata_offset is not None and conservative is not None:
        lead_cons = suricata_offset - conservative
        lead_opt = suricata_offset - optimistic

    print(f"\nflow_interval        {interval:.3f}s   window {infer.WINDOW}")
    print(f"netrikan forecast    window {fi}")
    print(f"  optimistic offset  "
          f"{f'{optimistic:.1f}s' if optimistic is not None else 'n/a'}"
          f"   (ignores the window fill -- do not quote this)")
    print(f"  conservative offset "
          f"{f'{conservative:.1f}s' if conservative is not None else 'n/a'}"
          f"   (after {infer.WINDOW} flows observed)")
    print(f"suricata first alert "
          f"{f'{suricata_offset:.1f}s' if suricata_offset is not None else 'n/a'}")
    if lead_cons is not None:
        verdict = ("netrikan first" if lead_cons > 0 else
                   "suricata first" if lead_cons < 0 else "tie")
        print(f"\nLEAD (conservative)  {lead_cons:+.1f}s   -> {verdict}")

    payload = {
        'pcap': os.path.abspath(args.pcap),
        'n_flows': int(len(df)),
        'n_windows': int(r['n_windows']),
        'window': int(infer.WINDOW),
        'flow_interval_s': round(float(interval), 4),
        'horizon_s': float(args.horizon),
        'rollout_source': r['rollout_source'],
        'capture_t0': str(t0),
        'suricata': {
            'available': True,
            'n_alerts': n_alerts,
            'first_alert_timestamp': (first_alert['timestamp']
                                      if first_alert else None),
            'first_alert_signature': (first_alert['alert']['signature']
                                      if first_alert else None),
            'first_alert_offset_s': (round(suricata_offset, 3)
                                     if suricata_offset is not None else None),
        },
        'netrikan': {
            'forecast_idx': None if fi is None else int(fi),
            'forecast_offset_s_optimistic': (round(optimistic, 3)
                                             if optimistic is not None else None),
            'forecast_offset_s_conservative': (round(conservative, 3)
                                               if conservative is not None else None),
            'self_reported_lead_seconds': r['lead_seconds'],
        },
        'lead_seconds_conservative': (round(lead_cons, 3)
                                      if lead_cons is not None else None),
        'lead_seconds_optimistic': (round(lead_opt, 3)
                                    if lead_opt is not None else None),
        'note': ('lead_seconds_conservative is the defensible number: it charges '
                 'Netrikan for the WINDOW flows it must observe before it can '
                 'forecast at all. Positive means Netrikan warned first. '
                 'netrikan.self_reported_lead_seconds is the old self-referential '
                 'figure, kept only for comparison -- it comes from the model\'s '
                 'own detection threshold, not from Suricata.'),
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved -> {args.out}")
    if tmp:
        print(f"suricata log kept at {tmp}")


if __name__ == '__main__':
    main()
