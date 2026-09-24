import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel
from attck_map import N_STAGES, MODEL_TO_CHAIN, BENIGN, DAMAGE_STAGES
import signals as sig
import forecast as fc
from pcap_ingest import PACKET_FEATURES

FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Fwd IAT Mean', 'Bwd IAT Mean',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt',
    'Pkt Len Var', 'Active Mean', 'Idle Mean',
]

WINDOW = 30
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, 'models', 'base_w30.pt')
WORLD_PATH = os.path.join(_ROOT, 'models', 'world_w30.pt')

# Prefer the copy shipped in models/ so a fresh clone works without the large
# processed dataset; fall back to the build directory when it is present.
_S1 = os.path.join(_ROOT, 'models', 'scaler_w30.pkl')
_S2 = os.path.join(_ROOT, 'data', 'processed', 'scaler.pkl')
SCALER_PATH = _S1 if os.path.exists(_S1) else _S2

# Temporal model vs. rule-based signal evidence. 0.30/0.70 split measured on
# validation set; run the ablation in baseline.py to update if retraining.
MODEL_WEIGHT = 0.30

_model = None
_world = None
_scaler = None
_device = torch.device('cpu')


def model_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def _load_model():
    global _model
    if _model is None:
        m = WorldModel()
        # strict=False: checkpoints trained before the world-model state head
        # carry no state_head weights. Classification is unaffected.
        m.load_state_dict(torch.load(MODEL_PATH, map_location=_device,
                                     weights_only=True), strict=False)
        m.eval()
        _model = m
    return _model


def _load_world():
    global _world
    if _world is None and os.path.exists(WORLD_PATH):
        m = WorldModel()
        m.load_state_dict(torch.load(WORLD_PATH, map_location=_device,
                                     weights_only=True), strict=False)
        m.eval()
        _world = m
    return _world


def _rollout_to_chain(rollout_stages):
    n, steps, _ = rollout_stages.shape
    chain = np.zeros((n, steps, N_STAGES))
    for mi, ci in MODEL_TO_CHAIN.items():
        chain[:, :, ci] = rollout_stages[:, :, mi]
    chain /= chain.sum(axis=2, keepdims=True) + 1e-12
    return chain


def _load_scaler():
    global _scaler
    if _scaler is None:
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
    return _scaler


def _flow_interval(df) -> float:
    if 'Timestamp' not in df.columns:
        return fc.DEFAULT_FLOW_INTERVAL
    try:
        ts = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        if ts.isna().all():
            ts = pd.to_datetime(df['Timestamp'], errors='coerce')
        d = ts.diff().dt.total_seconds().dropna()
        d = d[(d > 0) & (d < 300)]
        return float(d.median()) if len(d) else fc.DEFAULT_FLOW_INTERVAL
    except Exception:
        return fc.DEFAULT_FLOW_INTERVAL


def _fuse(model_probs5, signal):
    """Combine the temporal model's stage distribution with rule-based signal evidence."""
    p = np.zeros(N_STAGES)
    for mi, ci in MODEL_TO_CHAIN.items():
        p[ci] += float(model_probs5[mi])

    ev = np.full(N_STAGES, 0.02)
    s = signal["score"]
    if s > 0:
        ev[signal["stage_hint"]] += s
    else:
        ev[BENIGN] += 0.9
    ev = ev / ev.sum()

    fused = MODEL_WEIGHT * p + (1.0 - MODEL_WEIGHT) * ev
    return fused / fused.sum()


def analyze(df: pd.DataFrame, horizon_seconds: float = fc.HORIZON_SECONDS) -> dict:
    scaler = _load_scaler()
    model = _load_model()

    raw = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    scaled = scaler.transform(raw.astype(np.float32))
    ports = df['Dst Port'].values if 'Dst Port' in df.columns else None

    # packet-level columns exist only for PCAP input; CIC-IDS CSVs are flow records
    pkt_cols = [c for c in PACKET_FEATURES if c in df.columns]
    pkt_raw = df[pkt_cols].fillna(0.0).values.astype(np.float64) if pkt_cols else None

    n = len(scaled) - WINDOW
    if n <= 0:
        return None

    X = np.stack([scaled[i:i + WINDOW] for i in range(n)]).astype(np.float32)

    t = torch.tensor(X)
    with torch.no_grad():
        logits, breach, _ = model(t)
        model_probs = torch.softmax(logits, dim=1).numpy()
    breach = breach.numpy()

    # integrated gradients (gradient × input) on the breach head — not SHAP
    t2 = torch.tensor(X, requires_grad=True)
    _, b2, _ns = model(t2)
    b2.sum().backward()
    attribution = (t2.grad.detach().numpy() * X).mean(axis=1)

    interval = _flow_interval(df)
    steps = fc._steps_for_horizon(interval, WINDOW, horizon_seconds)
    world = _load_world()
    rollout_chain = None
    if world is not None:
        _, rs = world.rollout(t, steps)
        rollout_chain = _rollout_to_chain(rs.numpy())

    windows = []
    stage_prob_series = []
    for i in range(n):
        rawwin = raw[i:i + WINDOW]
        pw = ports[i:i + WINDOW] if ports is not None else None
        pkt = (dict(zip(pkt_cols, pkt_raw[i:i + WINDOW].mean(axis=0)))
               if pkt_raw is not None else None)
        s = sig.detect(rawwin, pw, pkt)
        fused = _fuse(model_probs[i], s)
        projs = list(rollout_chain[i]) if rollout_chain is not None else None
        f = fc.forecast(fused, interval, WINDOW, horizon_seconds, projections=projs)
        stage_prob_series.append(fused)
        windows.append({
            "idx": i,
            "signal": s,
            "stage_probs": fused,
            "stage": f["current_stage"],
            "forecast": f,
            "breach": float(breach[i]),
            "attribution": attribution[i],
            "time": i * interval,
        })

    stage_ids = np.array([w["stage"] for w in windows])
    forecast_idx = fc.first_forecast_index(stage_prob_series, 0.45, interval,
                                           horizon_seconds=horizon_seconds)
    detect_idx = fc.first_detection_index(stage_ids)

    lead = None
    if forecast_idx is not None and detect_idx is not None and detect_idx > forecast_idx:
        lead = (detect_idx - forecast_idx) * interval

    return {
        "windows": windows,
        "stage_ids": stage_ids,
        "breach": np.array([w["breach"] for w in windows]),
        "risk": np.array([w["forecast"]["damage_risk"] for w in windows]),
        "n_windows": n,
        "flow_interval": interval,
        "forecast_idx": forecast_idx,
        "detect_idx": detect_idx,
        "lead_seconds": lead,
        "n_flows": len(df),
    }
