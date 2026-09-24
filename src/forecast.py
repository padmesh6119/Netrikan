import numpy as np
from attck_map import (
    BENIGN, INITIAL_ACCESS, DOS, LATERAL, C2, EXFIL,
    N_STAGES, DAMAGE_STAGES, short,
)

# Stage persistence measured from DAPT 2020 (86,691 labelled flows, 2019 APT
# campaign): P(next flow is same stage | currently in stage). Attackers dwell in
# a phase far longer than doctrine suggests -- an earlier hand-written guess of
# 0.43 for Initial Access and Lateral was off by roughly 10x, which made the
# forecast escalate far too eagerly.
MEASURED_PERSISTENCE = {
    BENIGN: 0.980,          # DAPT Benign
    INITIAL_ACCESS: 0.929,  # DAPT Reconnaissance
    LATERAL: 0.931,         # DAPT Lateral Movement
    C2: 0.968,              # DAPT Establish Foothold
    DOS: 0.900,             # not present in DAPT -- volumetric floods sustain
    EXFIL: 0.930,           # DAPT Exfiltration had only 15 flows; use Lateral
}

# Relative weights for where a stage goes WHEN it moves. DAPT contains a single
# campaign (7 stage changes total), too few to estimate these, so the ordering
# stays ATT&CK doctrine: foothold -> lateral -> C2 -> exfil. Each row is scaled
# to fill exactly (1 - persistence), so the measured dwell is preserved.
DOCTRINE_SHAPE = np.array([
    # Ben   IA    DoS   Lat   C2    Exf
    [0.00, 0.55, 0.25, 0.12, 0.05, 0.03],  # from Benign
    [0.21, 0.00, 0.09, 0.54, 0.14, 0.02],  # from Initial Access
    [0.78, 0.11, 0.00, 0.07, 0.03, 0.01],  # from DoS
    [0.12, 0.09, 0.04, 0.00, 0.61, 0.14],  # from Lateral
    [0.09, 0.04, 0.03, 0.24, 0.00, 0.60],  # from C2
    [0.17, 0.03, 0.03, 0.11, 0.66, 0.00],  # from Exfil
])


def _build_transition():
    n = N_STAGES
    T = np.zeros((n, n))
    for i in range(n):
        p = MEASURED_PERSISTENCE[i]
        row = DOCTRINE_SHAPE[i].copy()
        row[i] = 0.0
        s = row.sum()
        T[i] = row / s * (1.0 - p) if s > 0 else 0.0
        T[i, i] = p
    return T / T.sum(axis=1, keepdims=True)


TRANSITION = _build_transition()

# position along the kill chain; DoS is an off-chain impact branch
CHAIN_POS = {BENIGN: 0, INITIAL_ACCESS: 1, LATERAL: 2, C2: 3, EXFIL: 4, DOS: 99}

DEFAULT_FLOW_INTERVAL = 1.2   # seconds per flow when timestamps absent
HORIZON_SECONDS = 900
MAX_STEPS = 400

# DAPT's campaign advanced roughly one phase per day, so a 60s horizon is right
# for volumetric attacks but far too short to see an intrusion escalate.
HORIZONS = {
    "60 seconds": 60,
    "15 minutes": 900,
    "1 hour": 3600,
    "6 hours": 21600,
    "24 hours": 86400,
}


def _steps_for_horizon(flow_interval: float, window: int = 10,
                       horizon_seconds: float = HORIZON_SECONDS) -> int:
    step_seconds = max(flow_interval, 0.05) * window
    return max(1, min(int(round(horizon_seconds / step_seconds)), MAX_STEPS))


def project(current: np.ndarray, steps: int):
    """Roll the stage distribution forward. Returns list of (N_STAGES,) arrays."""
    out, p = [], np.asarray(current, dtype=np.float64).copy()
    p = p / (p.sum() + 1e-12)
    for _ in range(steps):
        p = p @ TRANSITION
        p = p / (p.sum() + 1e-12)
        out.append(p.copy())
    return out


def _cap(s: str) -> str:
    return s if any(c.isupper() for c in s) else s[0].upper() + s[1:]


def _peaks(projections):
    """Highest probability each stage reaches anywhere in the horizon."""
    stack = np.vstack(projections)
    return stack.max(axis=0)


def _trajectory(projections, current_stage, start, peaks):
    """Immediate next stage plus the eventual destination along the kill chain."""
    # A Markov chain converges to its stationary distribution no matter where it
    # starts, and C2 holds the largest stationary mass (it has the highest
    # persistence). Over a long horizon that made benign traffic "predict" C2 --
    # the projection was forgetting its starting point. Only project a trajectory
    # when the current evidence actually indicates an attack.
    if current_stage == BENIGN and start[BENIGN] >= 0.5:
        return []

    cur = CHAIN_POS.get(current_stage, 0)

    # stages ahead on the chain count on reachability alone — they are where the
    # attacker goes next, whether or not their probability is still climbing
    onchain = sorted([s for s in range(N_STAGES)
                      if s != BENIGN and CHAIN_POS[s] != 99
                      and CHAIN_POS[s] > cur and peaks[s] >= 0.12],
                     key=lambda s: CHAIN_POS[s])
    if onchain:
        return [onchain[0], onchain[-1]] if len(onchain) > 1 else onchain

    # off-chain impact (DoS) must actually be rising to be worth forecasting
    offchain = [s for s in range(N_STAGES)
                if CHAIN_POS.get(s) == 99 and s != current_stage
                and peaks[s] >= 0.12 and peaks[s] > start[s] + 0.005]
    if offchain:
        return offchain[:1]

    # nothing further along the chain — attacker is already at the terminal stage
    return []


def _phrase(seq, current_stage):
    if not seq:
        return "No escalation expected" if current_stage == BENIGN \
            else f"Sustained {short(current_stage)}"
    if len(seq) == 1:
        return _cap(short(seq[0]))
    return f"{_cap(short(seq[0]))}, then {short(seq[1])}"


def forecast(stage_probs: np.ndarray, flow_interval: float = DEFAULT_FLOW_INTERVAL,
             window: int = 10, horizon_seconds: float = HORIZON_SECONDS,
             projections=None) -> dict:
    """
    stage_probs: (N_STAGES,) current distribution from the fusion layer.
    Returns the predicted trajectory over `horizon_seconds`, confidence, and ETA.
    """
    p = np.asarray(stage_probs, dtype=np.float64)
    if p.shape[0] < N_STAGES:
        p = np.pad(p, (0, N_STAGES - p.shape[0]))
    p = p / (p.sum() + 1e-12)

    current_stage = int(np.argmax(p))
    steps = _steps_for_horizon(flow_interval, window, horizon_seconds)
    if projections is None:
        projections = project(p, steps)
    else:
        projections = [np.asarray(pr, dtype=np.float64) for pr in projections]
        steps = len(projections)
    peaks = _peaks(projections)
    end = projections[-1]

    seq = _trajectory(projections, current_stage, p, peaks)
    phrase = _phrase(seq, current_stage)

    # risk that the trajectory reaches a damage stage within the horizon
    damage_risk, damage_step = 0.0, None
    step_seconds = max(flow_interval, 0.05) * window
    for i, proj in enumerate(projections):
        r = float(sum(proj[s] for s in DAMAGE_STAGES))
        damage_risk = max(damage_risk, r)
        if damage_step is None and r >= 0.45:
            damage_step = i + 1

    if seq:
        # how much of the plausible attack mass sits on the path we predicted
        nb_peaks = sorted((peaks[s] for s in range(N_STAGES) if s != BENIGN), reverse=True)
        best = float(sum(nb_peaks[:len(seq)]))
        concentration = float(sum(peaks[s] for s in seq)) / (best + 1e-12)
        confidence = 0.5 * (1.0 - float(end[BENIGN])) + 0.5 * concentration
    else:
        # No escalation predicted, so confidence is how sure we are of the state
        # we are in NOW. Reading it at the horizon end would measure chain
        # convergence rather than evidence.
        confidence = max(float(p[current_stage]), float(p[BENIGN]))

    return {
        "phrase": phrase,
        "sequence": seq,
        "current_stage": current_stage,
        "confidence": float(np.clip(confidence, 0.0, 0.99)),
        "damage_risk": damage_risk,
        "eta_seconds": int(damage_step * step_seconds) if damage_step else None,
        "steps": steps,
        "horizon_seconds": int(steps * step_seconds),
        "projections": projections,
    }


def first_forecast_index(stage_prob_series, threshold=0.45,
                         flow_interval=DEFAULT_FLOW_INTERVAL, sustain=3,
                         horizon_seconds=HORIZON_SECONDS):
    """Earliest window where we predict escalation before a damage stage is observable."""
    run = 0
    for i, p in enumerate(stage_prob_series):
        f = forecast(p, flow_interval, 10, horizon_seconds)
        if f["damage_risk"] >= threshold and f["current_stage"] not in DAMAGE_STAGES:
            run += 1
            if run >= sustain:
                return i - sustain + 1
        else:
            run = 0
    return None


def first_detection_index(stage_ids, sustain=3):
    """Earliest window the fused model output enters a damage stage (detection baseline).
    This uses the model's own stage output — not an external IDS."""
    run = 0
    for i, s in enumerate(stage_ids):
        if int(s) in DAMAGE_STAGES:
            run += 1
            if run >= sustain:
                return i - sustain + 1
        else:
            run = 0
    return None
