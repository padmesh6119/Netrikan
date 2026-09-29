import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel, ONSET_HORIZONS
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

# Deployed checkpoint: cic_v2_w30 is the consolidated model — trained on all 7
# time-sorted CIC-IDS-2018 days, all 5 classes, with the state head (rollout),
# onset head (forecasting) and temporal attention (explainability) all trained,
# selected on a combined onset+classification criterion. One checkpoint that does
# everything. Falls back to base_w30 (classifier only) if it is not present.
_M1 = os.path.join(_ROOT, 'models', 'cic_v2_w30.pt')
_M2 = os.path.join(_ROOT, 'models', 'base_w30.pt')
MODEL_PATH = _M1 if os.path.exists(_M1) else _M2

# The rollout uses the same checkpoint when it carries a trained state head
# (cic_v2 does); world_w30.pt remains an optional dedicated override.
WORLD_PATH = os.path.join(_ROOT, 'models', 'world_w30.pt')

# Scaler must match the deployed checkpoint. cic_v2 was fit on the 7-day build;
# base_w30 on its own. Pick the scaler that goes with MODEL_PATH.
if MODEL_PATH == _M1 and os.path.exists(os.path.join(_ROOT, 'models', 'scaler_cic_full.pkl')):
    SCALER_PATH = os.path.join(_ROOT, 'models', 'scaler_cic_full.pkl')
else:
    _S1 = os.path.join(_ROOT, 'models', 'scaler_w30.pkl')
    _S2 = os.path.join(_ROOT, 'data', 'processed', 'scaler.pkl')
    SCALER_PATH = _S1 if os.path.exists(_S1) else _S2

# Temporal model vs. rule-based signal evidence, in the fused stage distribution.
# MEASURED, not guessed: bench/model_weight_ablation.py sweeps it on DAPT 2020
# (models/model_weight_ablation_cic_v2_w30.json). Cross-dataset binary SEDI rises
# monotonically with model weight — 0.0: -0.256, 0.5: -0.102, 0.7: +0.333,
# 0.9: +0.405, 1.0: +0.416 — because the 14 port-based rules in signals.py encode
# CIC's environment and do not transfer to unseen networks. 0.9 captures almost
# all of the cross-dataset gain (0.405 vs 1.0's 0.416) while keeping a 10% rule
# contribution as a familiar-network tiebreaker. Override for a network unlike CIC
# via NETRIKAN_MODEL_WEIGHT (e.g. 1.0 to drop the rules entirely). Signal names and
# evidence strings still display regardless of this weight — it only sets the
# fusion mix, not whether a rule fires.
def _model_weight():
    v = os.environ.get('NETRIKAN_MODEL_WEIGHT')
    if v is not None:
        try:
            return min(max(float(v), 0.0), 1.0)
        except ValueError:
            pass
    return 0.90


MODEL_WEIGHT = _model_weight()

_model = None
_world = None
_scaler = None
_device = torch.device('cpu')

# Calibration temperature. Preferred: the held-out sidecar models/<ckpt>_calib.json
# from `calibration.py --heldout`, which carries a ship gate -- when "deploy" is
# false its temperature is 1.0 and the raw softmax is used. Fallback for
# checkpoints without a sidecar: models/temperature.json (val-only fit).
CALIB_PATH = os.path.splitext(MODEL_PATH)[0] + '_calib.json'
TEMPERATURE_PATH = os.path.join(_ROOT, 'models', 'temperature.json')
_T = 1.0


def model_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def _load_model():
    global _model, _T
    if _model is None:
        sd = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        # attention is architectural, so it must match the checkpoint exactly:
        # building the model without it would leave the weights unused and
        # change every prediction
        m = WorldModel(attention='attn.weight' in sd)
        # strict=False: checkpoints trained before the state and onset heads
        # existed carry no weights for them. Classification is unaffected, but
        # an untrained onset head outputs noise -- onset_trained() reports that.
        m.load_state_dict(sd, strict=False)
        m.eval()
        _model = m
        if os.path.exists(CALIB_PATH):
            with open(CALIB_PATH) as f:
                _T = float(json.load(f)['temperature'])
        elif os.path.exists(TEMPERATURE_PATH):
            with open(TEMPERATURE_PATH) as f:
                _T = float(json.load(f)['T'])
    return _model


def _ckpt_keys() -> set:
    return set(torch.load(MODEL_PATH, map_location=_device, weights_only=True))


def onset_trained() -> bool:
    """True when MODEL_PATH actually contains trained onset_head weights.

    A checkpoint from before the onset head existed loads fine under
    strict=False, but its onset outputs are randomly initialised noise. Callers
    must not present those as forecasts.
    """
    return 'onset_head.weight' in _ckpt_keys()


def attention_trained() -> bool:
    """True when the checkpoint was trained with temporal attention, so the
    per-timestep weights genuinely drive the prediction and can be shown as an
    explanation."""
    return 'attn.weight' in _ckpt_keys()


def world_model_ready() -> bool:
    """True when the trained dynamics checkpoint is present.

    A learned rollout is available when either a dedicated WORLD_PATH exists or
    the deployed MODEL_PATH itself carries a trained state head. Only when neither
    holds does forecast() fall back to the hand-coded DOCTRINE_SHAPE matrix.
    """
    if os.path.exists(WORLD_PATH):
        return True
    return 'state_head.weight' in _ckpt_keys()


def _load_world():
    global _world
    if _world is None:
        # a dedicated world checkpoint wins; otherwise reuse the main model if it
        # has a trained state head (cic_v2 does), so no separate file is needed
        if os.path.exists(WORLD_PATH):
            sd = torch.load(WORLD_PATH, map_location=_device, weights_only=True)
        elif 'state_head.weight' in _ckpt_keys():
            return _load_model()
        else:
            print("WARNING: no trained state head available — K-step rollout is "
                  "running on the hand-coded Markov fallback, not learned dynamics.",
                  file=sys.stderr)
            return None
        m = WorldModel(attention='attn.weight' in sd)
        m.load_state_dict(sd, strict=False)
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
        logits, breach, _, onset_logits, attn = model(t, return_attention=True)
        model_probs = torch.softmax(logits / _T, dim=1).numpy()
        onset = torch.sigmoid(onset_logits).numpy()
    breach = breach.numpy()
    attn = attn.numpy() if attn is not None else None

    # gradient × input on the breach head — feature attribution, not SHAP
    t2 = torch.tensor(X, requires_grad=True)
    _, b2, _ns, _onset = model(t2)
    b2.sum().backward()
    attribution = (t2.grad.detach().numpy() * X).mean(axis=1)

    # an onset head that was never trained emits noise; do not surface it
    has_onset = onset_trained()
    if not has_onset:
        onset = None

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
            "onset": (dict(zip(ONSET_HORIZONS, onset[i].tolist()))
                      if onset is not None else None),
            # per-timestep attention over the window: which flow in the window
            # the model weighted most. None when the checkpoint has no attention.
            "attention": attn[i].tolist() if attn is not None else None,
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
        "rollout_source": "learned" if world is not None else "markov_fallback",
        "onset_horizons": list(ONSET_HORIZONS) if has_onset else None,
        "temperature": _T,
        # explainability provenance, so the UI can name the method honestly
        "attribution_method": "gradient x input",
        "has_attention": attn is not None,
    }


_shap_explainer = None
_shap_bg_key = None


def shap_for_windows(df, window_indices, target="breach", top_k=5, nsamples=64):
    """Real SHAP (shap.GradientExplainer) for specific windows of a capture.

    On demand only: SHAP runs many samples per input, so this is for a handful of
    windows the analyst selects, not a whole capture. The always-on gradient x
    input in analyze() covers the per-window case cheaply.

    Background is drawn from the lowest-breach (most benign-looking) windows of the
    same capture, so attributions are relative to this network's normal traffic.

    Returns one dict per requested index: {idx, target, top_features,
    timestep_profile}. top_features are (feature, shap_value, direction), summed
    over the window; timestep_profile is |shap| per flow in the window.
    """
    global _shap_explainer, _shap_bg_key
    from explain_shap import ShapExplainer, build_background

    scaler = _load_scaler()
    model = _load_model()
    raw = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    scaled = scaler.transform(raw.astype(np.float32))
    n = len(scaled) - WINDOW
    if n <= 0:
        return []
    X = np.stack([scaled[i:i + WINDOW] for i in range(n)]).astype(np.float32)

    # background = most-benign windows by breach score, cached per capture shape
    key = (id(df), X.shape, target)
    if _shap_explainer is None or _shap_bg_key != key:
        with torch.no_grad():
            b = model(torch.tensor(X))[1].numpy()
        benign = X[np.argsort(b)[:min(200, len(X))]]
        bg = build_background(benign, n=64)
        _shap_explainer = ShapExplainer(model, bg,
                                        k_index=(ONSET_HORIZONS.index(5)
                                                 if 5 in ONSET_HORIZONS else 0))
        _shap_bg_key = key

    idxs = [i for i in window_indices if 0 <= i < n]
    if not idxs:
        return []
    sv = _shap_explainer.explain(X[idxs], target=target, nsamples=nsamples)
    out = []
    for row, i in enumerate(idxs):
        prof = np.abs(sv[row]).sum(axis=1)
        out.append({
            "idx": int(i),
            "target": target,
            "method": "shap.GradientExplainer (expected gradients)",
            "top_features": _shap_explainer.top_features(sv[row], FEATURES, top_k),
            "timestep_profile": [round(float(v), 5) for v in prof],
            "peak_step": int(np.argmax(prof)),
        })
    return out


def explain(window: dict, top_k: int = 5, shap_result: dict = None) -> dict:
    """Structured explanation of one window's forecast, from analyze()'s output.

    Combines the four evidence channels the model actually uses, each labelled by
    where it comes from so nothing is presented as more than it is:

      features   gradient x input on the breach head -- which of the 24 flow
                 features pushed the malicious score up or down
      timing     temporal attention -- which flow within the window the model
                 weighted most (only when the checkpoint was trained with it)
      rule       which of the 14 hand-written detectors fired, and its evidence
      forecast   the predicted trajectory, damage risk, and per-horizon onset
                 probability (only when the onset head is trained)

    Returns a dict with those parts plus a one-line human-readable `summary`.
    """
    from attck_map import short, get_stage

    # Feature attribution: prefer real SHAP when it was computed for this window,
    # fall back to the always-on gradient x input otherwise. Labelled either way.
    if shap_result is not None:
        features = [{"feature": d["feature"], "contribution": d["shap_value"],
                     "direction": d["direction"]}
                    for d in shap_result["top_features"]]
        attribution_method = shap_result["method"]
    else:
        attr = np.asarray(window["attribution"])
        order = np.argsort(-np.abs(attr))[:top_k]
        features = [{"feature": FEATURES[j], "contribution": round(float(attr[j]), 5),
                     "direction": "raises" if attr[j] > 0 else "lowers"}
                    for j in order]
        attribution_method = "gradient x input"

    timing = None
    if window.get("attention") is not None:
        a = np.asarray(window["attention"])
        peak = int(np.argmax(a))
        timing = {"peak_step": peak, "peak_weight": round(float(a[peak]), 4),
                  "position": ("most recent flow" if peak >= len(a) - 3
                               else "oldest flow" if peak <= 2
                               else f"flow {peak+1} of {len(a)} in the window")}

    sig_ = window["signal"]
    rule = {"name": sig_["name"], "evidence": sig_.get("evidence", ""),
            "score": round(float(sig_.get("score", 0.0)), 3),
            "fired": float(sig_.get("score", 0.0)) > 0}

    f = window["forecast"]
    forecast = {"phrase": f["phrase"],
                "trajectory": [short(s) for s in f.get("sequence", [])],
                "damage_risk": round(float(f["damage_risk"]), 4),
                "confidence": round(float(f["confidence"]), 4),
                "eta_seconds": f.get("eta_seconds"),
                "onset_probability": ({f"k{k}": round(float(v), 4)
                                       for k, v in window["onset"].items()}
                                      if window.get("onset") else None)}

    stage = get_stage(window["stage"])
    top_feat = features[0]["feature"] if features else "n/a"
    bits = [f"Stage now: {stage['name']}"]
    if rule["fired"]:
        bits.append(f"rule '{rule['name']}' fired ({rule['evidence']})")
    bits.append(f"top feature: {top_feat} {features[0]['direction']} risk"
                if features else "")
    if timing:
        bits.append(f"attention on the {timing['position']}")
    if forecast["onset_probability"]:
        bits.append(f"P(stage change ≤5 windows)="
                    f"{forecast['onset_probability'].get('k5', '?')}")
    bits.append(f"forecast: {forecast['phrase']} "
                f"(risk {forecast['damage_risk']:.0%})")

    return {
        "window_idx": window["idx"],
        "stage": stage["name"],
        "mitre": stage.get("mitre", ""),
        "breach_probability": round(float(window["breach"]), 4),
        "attribution_method": attribution_method,
        "features": features,
        "timing": timing,
        "rule": rule,
        "forecast": forecast,
        "summary": " · ".join(b for b in bits if b),
    }
