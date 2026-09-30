"""Response layer: turns forecast alerts into incidents and attaches MITRE D3FEND countermeasures.

An incident is a run of alert minutes on one host (one host + technique on ZeekData24), with alert gaps of up to
K minutes merged. The playbook maps ATT&CK tactics/techniques to D3FEND countermeasures; the concrete actions are
filled in from the flows around the incident (which peers and ports to filter). Rules are templates for an
analyst to review, never applied automatically.

Honesty rules: on DAPT2020 the stage model is at chance across held-out days, so priority there comes from the
risk score and duration only and the stage-specific playbook is labelled low-confidence. On ZeekData24 the
technique comes from the recognizer (near-perfect in-corpus) and is used for priority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from netrikan import infer
from netrikan.dapt import STAGE_NAMES
from netrikan.features import PRIVATE
from netrikan.zeek import TECH_NAME, TECH_TACTIC, TECHS

from . import dapt, z24
from .common import bucket_ms, ms, out

router = APIRouter(prefix="/api/response")

# ---------------------------------------------------------------- playbook (ATT&CK -> D3FEND)
# Each countermeasure: D3FEND technique name, its D3FEND tactic, the action (templated), and why it applies.
CM = dict


FIRST_RESPONSE = [
    CM(d3fend="Network Traffic Community Deviation", tactic="Detect",
       action="Compare {host}'s peers and ports in the incident window against its usual community; confirm the deviation the model scored.",
       why="Stage-agnostic triage: verifies the behaviour change behind the risk score before acting."),
    CM(d3fend="Network Traffic Filtering", tactic="Isolate",
       action="If confirmed, move {host} to a quarantine segment or apply a host ACL allowing only management traffic.",
       why="Containment that does not depend on knowing the exact stage."),
]

BY_TACTIC = {
    "Reconnaissance": [
        CM(d3fend="Connection Attempt Analysis", tactic="Detect", action="Review connection attempts from {src} across {n_peers} peers / {n_ports} ports; confirm fan-out scanning.",
           why="Active scanning shows as many short or failed connection attempts to many ports."),
        CM(d3fend="Inbound Traffic Filtering", tactic="Isolate", action="Block {src} at the segment or perimeter ACL.", why="Stops the scan before it finds an exposed service."),
        CM(d3fend="Decoy Network Resource", tactic="Deceive", action="Expose a decoy service on a scanned port to confirm intent and fingerprint the scanner.",
           why="Scanners touch decoys early; any interaction is a high-confidence signal."),
    ],
    "Initial Access": [
        CM(d3fend="Inbound Traffic Filtering", tactic="Isolate", action="Restrict ports {ports} on {dst} to known sources; drop traffic from {src}.",
           why="Limits exploitation of the exposed service while it is investigated."),
        CM(d3fend="Software Update", tactic="Harden", action="Check the services on {dst} ports {ports} for missing patches and update them.",
           why="Exploitation of public-facing applications relies on unpatched software."),
        CM(d3fend="Protocol Metadata Anomaly Detection", tactic="Detect", action="Inspect protocol metadata of sessions to {dst}:{ports} for malformed or unusual requests.",
           why="Exploit payloads often produce abnormal protocol fields."),
    ],
    "Valid Accounts": [
        CM(d3fend="Remote Terminal Session Detection", tactic="Detect", action="List interactive sessions from {src} to {peers}; verify each against expected administrators.",
           why="Stolen credentials are used through legitimate remote-access channels."),
        CM(d3fend="Account Locking", tactic="Evict", action="Lock or reset the accounts used from {src} in this window.", why="Removes the attacker's access path."),
        CM(d3fend="Multi-factor Authentication", tactic="Harden", action="Require MFA for remote access to {peers}.", why="Valid passwords alone stop being sufficient."),
    ],
    "Credential Access": [
        CM(d3fend="Authentication Event Thresholding", tactic="Detect", action="Alert on failed-login bursts from {src} to {peers} on ports {ports}.",
           why="Brute force produces authentication failures far above the host's baseline."),
        CM(d3fend="Account Locking", tactic="Evict", action="Temporarily lock accounts targeted from {src}.", why="Stops guessing against the targeted accounts."),
        CM(d3fend="Inbound Traffic Filtering", tactic="Isolate", action="Block {src} from reaching {peers} on ports {ports}.", why="Cuts the brute-force channel."),
        CM(d3fend="Strong Password Policy", tactic="Harden", action="Audit password strength for accounts on {peers}.", why="Reduces the chance a guess succeeds."),
    ],
    "Lateral Movement": [
        CM(d3fend="Remote Terminal Session Detection", tactic="Detect", action="Review remote sessions between {host} and the internal peers it reached: {int_peers}.",
           why="Lateral movement uses SSH/RDP/SMB sessions between internal hosts."),
        CM(d3fend="Administrative Network Activity Analysis", tactic="Detect", action="Check whether {host} normally makes administrative connections (remote-admin ports used here: {admin_ports}).",
           why="A workstation suddenly acting like an admin host is a strong indicator."),
        CM(d3fend="Network Traffic Filtering", tactic="Isolate", action="Block east-west traffic from {host} to {int_peers} except required services.",
           why="Stops the spread while the foothold is removed."),
    ],
    "Exfiltration": [
        CM(d3fend="Per Host Download-Upload Ratio Analysis", tactic="Detect", action="Compare {host}'s outbound vs inbound bytes; top external destinations by bytes sent: {ext_peers}.",
           why="Exfiltration flips the usual download-heavy ratio."),
        CM(d3fend="Outbound Traffic Filtering", tactic="Isolate", action="Block outbound traffic from {host} to {ext_peers} once confirmed they are not shared infrastructure.", why="Stops data leaving while the incident is scoped."),
    ],
}
TECH_TACTIC_KEY = {"T1595": "Reconnaissance", "T1190": "Initial Access", "T1078": "Valid Accounts", "T1110": "Credential Access", "T1048": "Exfiltration"}
ADMIN_PORTS = {22, 23, 135, 139, 445, 3389, 5900, 5985, 5986}
TECH_SEV = {"T1048": 3, "T1190": 3, "T1078": 3, "T1110": 2, "T1595": 1}


def runs(alarm: np.ndarray, start: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """[a, b] row ranges of alert minutes, merging gaps of up to `gap` rows within one sequence."""
    idx = np.flatnonzero(alarm)
    out: list[list[int]] = []
    for i in idx:
        if out and start[i] == start[out[-1][1]] and i - out[-1][1] <= gap + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def targets(flows: pd.DataFrame, host: str, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    """Peers and ports the host talked to around the incident, for the concrete actions."""
    w = flows[(flows["ts"] >= t0) & (flows["ts"] < t1) & ((flows["Src IP"] == host) | (flows["Dst IP"] == host))]
    outb, inb = w[w["Src IP"] == host], w[w["Dst IP"] == host]
    top = lambda s, n=5: [{"value": str(k), "flows": int(v)} for k, v in s.value_counts().head(n).items()]
    port = lambda s: s.astype(int).astype(str)
    internal = outb["Dst IP"].astype(str).str.match(PRIVATE)
    ext_bytes = outb[~internal].groupby("Dst IP")["Total Length of Fwd Packet"].sum().sort_values(ascending=False)
    admin = sorted({int(p) for p in outb["Dst Port"].unique() if int(p) in ADMIN_PORTS})
    return {"flows": len(w), "out_peers": top(outb["Dst IP"]), "out_ports": top(port(outb["Dst Port"])),
            "internal_peers": top(outb.loc[internal, "Dst IP"]),
            "external_by_bytes": [{"value": str(k), "bytes": float(v)} for k, v in ext_bytes.head(5).items()],
            "admin_ports": admin,
            "in_peers": top(inb["Src IP"]), "in_ports": top(port(inb["Dst Port"])),
            "n_out_peers": int(outb["Dst IP"].nunique()), "n_out_ports": int(outb["Dst Port"].nunique()),
            "bytes_out": float(outb["Total Length of Fwd Packet"].sum() + inb["Total Length of Bwd Packet"].sum()),
            "bytes_in": float(outb["Total Length of Bwd Packet"].sum() + inb["Total Length of Fwd Packet"].sum())}


def fill(cms: list[dict], ctx: dict) -> list[dict]:
    return [{**c, "action": c["action"].format(**ctx)} for c in cms]


def rules(host: str, tg: dict, key: str | None) -> list[str]:
    """Example iptables rules (templates for review) for the containment step. Never applied by Netrikan."""
    r = [f"# quarantine {host}: drop everything except the admin subnet (set ADMIN_SUBNET first)",
         f"iptables -I FORWARD -s {host} -j DROP",
         f"iptables -I FORWARD -s {host} -d $ADMIN_SUBNET -j ACCEPT"]
    if key == "Lateral Movement" and tg["internal_peers"]:
        r.append("# block east-west traffic to the internal peers it reached")
        r += [f"iptables -I FORWARD -s {host} -d {p['value']} -j DROP" for p in tg["internal_peers"][:3]]
    if key == "Exfiltration" and tg["external_by_bytes"]:
        r.append("# block the external destinations that received the most bytes (check they are not shared infrastructure)")
        r += [f"iptables -I FORWARD -s {host} -d {p['value']} -j DROP" for p in tg["external_by_bytes"][:3]]
    if key in ("Reconnaissance", "Credential Access", "Initial Access", "Valid Accounts"):
        ports = [p["value"] for p in tg["out_ports"][:3]]
        r.append(f"# {host} is the source of the activity: reject it on the targeted ports")
        r += [f"iptables -I FORWARD -s {host} -p tcp --dport {p} -j REJECT" for p in ports]
    return r


def horizon_lead(alarm: np.ndarray, onset: int, start: int, K: int) -> int:
    """Warning lead as the evaluation measures it (metrics.lead_times): minutes between the earliest alert in the
    K minutes before the attack onset and the onset itself. 0 = no alert inside the horizon. Never exceeds K."""
    lo = max(int(start), onset - K)
    hit = np.flatnonzero(alarm[lo:onset])
    return int(onset - lo - hit[0]) if len(hit) else 0


def priority(score: float) -> str:
    return "P1" if score >= 0.7 else "P2" if score >= 0.4 else "P3"


# ---------------------------------------------------------------- DAPT2020
def dapt_incidents(source: str, model: str, thr: float):
    flows, _, tab, pr = dapt.run(source)
    p = pr[model]
    labelled = bool(flows["labelled"].any())
    T = infer.times(tab)
    out = []
    for a, b in runs(p >= thr, tab.start, tab.K):
        i = a + int(np.argmax(p[a:b + 1]))
        ratio = float(p[i] / thr)
        dur = b - a + 1
        score = min(1.0, 0.5 * min(ratio, 3) / 3 + 0.5 * min(dur, 10) / 10)
        sp = pr["stage"][i][1:]
        inc = {"id": f"{tab.host[a]}|{ms(T[a])}", "host": tab.host[a], "start": ms(T[a]), "end": ms(T[b]) + 60_000, "row": int(i),
               "peak": float(p[i]), "ratio": ratio, "minutes": dur, "priority": priority(score), "score": score,
               "tactic": STAGE_NAMES[1 + int(np.argmax(sp))], "tactic_p": float(sp.max()), "technique": None}
        inc["playbook_key"] = inc["tactic"]
        if labelled:
            e = min(b + tab.K, int(tab.end[a]) - 1)
            hit = np.flatnonzero(tab.attack[a:e + 1])
            inc["outcome"] = "attack" if len(hit) else "no_attack"
            # lead_min: warning inside the forecast horizon (<= K); run_min: first alert of the merged run to the attack
            inc["lead_min"] = horizon_lead(p >= thr, a + int(hit[0]), tab.start[a], tab.K) if len(hit) else None
            inc["run_min"] = int(hit[0]) if len(hit) else None
            inc["true_tactic"] = STAGE_NAMES[int(tab.stage[a + hit[0]])] if len(hit) else None
        out.append(inc)
    return out


# ---------------------------------------------------------------- ZeekData24
def z24_incidents(week: str, rung: str, budget: float):
    R = z24.analyse(week, 0.5)
    S, pr, fut = R["S"], R["pr"], R["fut"]
    if S is None:
        return []
    K = z24.mx()["dataset"]["horizon_K_minutes"]
    out = []
    for j, t in enumerate(TECHS):
        thr = z24.mx()["forecast"]["per_technique"][rung][t][f"thr@{budget}/h"]
        pj = pr[rung][:, j]
        for a, b in runs((pj >= thr) & ~S.F[:, j], S.start, K):
            i = a + int(np.argmax(pj[a:b + 1]))
            ratio = float(pj[i] / thr)
            score = min(1.0, TECH_SEV[t] / 3 * 0.5 + 0.5 * min(ratio, 3) / 3)
            e = min(b + K, int(S.end[a]) - 1)
            hit = np.flatnonzero(S.F[a:e + 1, j])
            recent = S.Fp[max(int(S.start[a]), a - 60):a + 1].any(0)
            out.append({"id": f"{S.host[a]}|{int(S.bucket[a]) * 60_000}|{t}", "host": S.host[a], "start": int(S.bucket[a]) * 60_000,
                        "end": int(S.bucket[b] + 1) * 60_000, "row": int(i), "peak": float(pj[i]), "ratio": ratio, "minutes": b - a + 1,
                        "priority": priority(score), "score": score, "tactic": TECH_TACTIC[t], "playbook_key": TECH_TACTIC_KEY[t],
                        "tactic_p": None, "technique": t,
                        "recent": [TECHS[k] for k in np.flatnonzero(recent)],
                        "outcome": "attack" if len(hit) else "no_attack", "true_tactic": None,
                        "lead_min": horizon_lead((pj >= thr) & ~S.F[:, j], a + int(hit[0]), S.start[a], K) if len(hit) else None,
                        "run_min": int(hit[0]) if len(hit) else None})
    out.sort(key=lambda r: r["start"])
    return out


@router.get("/incidents")
@out
def incidents(dataset: str, source: str, model: str = "lag", thr: float | None = None, rung: str = "sched", budget: float = 1.0):
    if dataset == "dapt":
        thr = thr if thr is not None else dapt.default_threshold(model)
        rows = dapt_incidents(source, model, thr)
        labelled = bool(dapt.run(source)[0]["labelled"].any())
        note = "Stage comes from a model at chance on held-out days; priority uses risk and duration only."
    elif dataset == "z24":
        if rung not in z24.RUNGS:
            raise HTTPException(400, "unknown model")
        rows = z24_incidents(source, rung, budget)
        labelled = True
        note = "Technique comes from the recognizer; priority weighs technique severity and forecast strength."
    else:
        raise HTTPException(404, "unknown dataset")
    return {"dataset": dataset, "source": source, "labelled": labelled, "note": note, "threshold": thr, "incidents": rows}


@router.get("/incident")
@out
def incident(dataset: str, source: str, id: str, model: str = "lag", thr: float | None = None, rung: str = "sched", budget: float = 1.0):
    """One incident with its evidence, D3FEND countermeasures and example containment rules."""
    rows = incidents.__wrapped__(dataset, source, model, thr, rung, budget)["incidents"]
    inc = next((r for r in rows if r["id"] == id), None)
    if inc is None:
        raise HTTPException(404, "incident not found")
    t0, t1 = pd.to_datetime(inc["start"], unit="ms"), pd.to_datetime(inc["end"], unit="ms")
    if dataset == "dapt":
        flows, model_name, tab, pr = dapt.run(source)
        drivers = [{"feature": n, "value": v} for n, v in infer.top_drivers(dapt.bundle(model_name), pr, inc["row"], model, 6)]
        K = tab.K
        confidence = {"level": "low", "text": f"Stage model accuracy on held-out days: {dapt.mx()['stage']['accuracy']:.0%} "
                                                f"(majority baseline {dapt.mx()['stage']['majority_baseline_accuracy']:.0%}). Treat the stage playbook as a hypothesis."}
        series_rows = np.flatnonzero((tab.host == inc["host"]) & (infer.times(tab) >= t0 - pd.Timedelta(minutes=30)) & (infer.times(tab) < t1 + pd.Timedelta(minutes=30)))
        series = {"t": bucket_ms(tab.bucket[series_rows]), "p": np.round(pr[model][series_rows], 4), "attack": tab.attack[series_rows].astype(int)}
    else:
        flows = z24.week_flows(source)
        R = z24.analyse(source, 0.5)
        S = R["S"]
        j = TECHS.index(inc["technique"])
        drivers = z24.forecast_explain.__wrapped__(source, inc["technique"], inc["row"])["drivers"]
        K = z24.mx()["dataset"]["horizon_K_minutes"]
        rec = z24.mx()["recognizer"]["per_technique"][inc["technique"]]
        confidence = {"level": "high", "text": f"Technique recognizer PR-AUC {rec['pr_auc']:.3f} on held-out weeks of the same scripted campaign "
                                                 "(in-corpus; not evidence about real-world attacks)."}
        a, b = int(S.start[inc["row"]]), int(S.end[inc["row"]])
        tt = S.bucket[a:b] * 60_000
        m = (tt >= inc["start"] - 30 * 60_000) & (tt < inc["end"] + 30 * 60_000)
        rr = np.arange(a, b)[m]
        series = {"t": tt[m], "p": np.round(R["pr"][rung][rr, j], 4), "attack": S.F[rr, j].astype(int)}
    tg = targets(flows, inc["host"], t0 - pd.Timedelta(minutes=5), t1 + pd.Timedelta(minutes=K))
    peers = [p["value"] for p in (tg["out_peers"] or tg["in_peers"])[:3]]
    ports = [p["value"] for p in (tg["out_ports"] or tg["in_ports"])[:3]]
    ipeers = [p["value"] for p in tg["internal_peers"][:3]]
    xpeers = [p["value"] for p in tg["external_by_bytes"][:3]]
    ctx = {"host": inc["host"], "src": inc["host"], "dst": ", ".join(peers) or "the targeted hosts", "peers": ", ".join(peers) or "its peers",
           "ports": ", ".join(ports) or "the observed ports", "n_peers": tg["n_out_peers"], "n_ports": tg["n_out_ports"],
           "int_peers": ", ".join(ipeers) or "(no internal peers in this window)", "ext_peers": ", ".join(xpeers) or "(no external peers in this window)",
           "admin_ports": ", ".join(map(str, tg["admin_ports"])) or "none observed in this window"}
    playbook = [{"group": "First response (stage-agnostic)", "confidence": "high", "items": fill(FIRST_RESPONSE, ctx)}]
    playbook.append({"group": f"{inc['tactic']}{' · ' + inc['technique'] + ' ' + TECH_NAME[inc['technique']] if inc['technique'] else ''}",
                     "confidence": confidence["level"], "items": fill(BY_TACTIC.get(inc["playbook_key"], []), ctx)})
    return {"incident": inc, "drivers": drivers, "targets": tg, "confidence": confidence, "playbook": playbook,
            "rules": rules(inc["host"], tg, inc["playbook_key"]), "series": series,
            "attack_ref": {"tactic": inc["tactic"], "technique": inc["technique"], "technique_name": TECH_NAME.get(inc["technique"] or "", None),
                           "technique_tactic": TECH_TACTIC.get(inc["technique"] or "", None)}}
