import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from netrikan import models  # noqa: E402
from netrikan.dapt import normalise  # noqa: E402
from netrikan.features import FEATURES, build_grid, make_table  # noqa: E402

FMT = "%d/%m/%Y %I:%M:%S %p"


def synth(n=600, seed=0, t_end_min=120):
    """Random flows for two internal hosts over `t_end_min` minutes, attack flows in bursts."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2019-07-16 12:00:00")
    ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, t_end_min * 60, n)), unit="s")
    src = rng.choice(["192.168.3.29", "192.168.3.30", "8.8.8.8"], n)
    dst = np.where(src == "8.8.8.8", rng.choice(["192.168.3.29", "192.168.3.30"], n), "8.8.8.8")
    minute = (ts - t0).total_seconds() // 60
    stage = np.where(((minute % 40) > 30) & (dst == "192.168.3.29"), 1, 0)
    df = pd.DataFrame({
        "Src IP": src, "Dst IP": dst, "Dst Port": rng.choice([22, 80, 443, 3306, 50000], n), "Protocol": 6,
        "Timestamp": ts.strftime(FMT), "Flow Duration": rng.integers(1, 5_000_000, n),
        "Total Fwd Packet": rng.integers(1, 50, n), "Total Bwd packets": rng.integers(0, 50, n),
        "Total Length of Fwd Packet": rng.integers(0, 9000, n), "Total Length of Bwd Packet": rng.integers(0, 9000, n),
        "SYN Flag Count": rng.integers(0, 3, n), "RST Flag Count": rng.integers(0, 2, n),
        "ACK Flag Count": rng.integers(0, 40, n), "PSH Flag Count": rng.integers(0, 10, n),
        "FIN Flag Count": rng.integers(0, 2, n), "Packet Length Mean": rng.uniform(0, 900, n),
        "Stage": np.where(stage > 0, "Reconnaissance", "Benign")})
    return normalise(df)


def test_features_are_causal():
    """Corrupting every flow after bucket T must not change any row at or before T."""
    df = synth()
    cut = df["ts"].iloc[0] + pd.Timedelta(minutes=60)
    fut = df["ts"] >= cut
    bad = df.copy()
    bad.loc[fut, ["Total Fwd Packet", "SYN Flag Count", "Dst Port"]] = [9999, 99, 22]
    bad.loc[fut, "Stage"] = "Data Exfiltration"
    bad["stage"] = np.where(fut, 4, bad["stage"])
    a, b = build_grid(df, 60), build_grid(bad, 60)
    cutb = int((cut - pd.Timestamp("1970-01-01")).total_seconds() // 60)
    a, b = a[a.bucket < cutb].reset_index(drop=True), b[b.bucket < cutb].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[FEATURES + ["bucket", "host"]], b[FEATURES + ["bucket", "host"]])


def test_history_never_reaches_forward_or_across_sequences():
    tab = make_table(build_grid(synth(), 60), K=5, L=10)
    i = np.arange(tab.n)[:, None]
    real = tab.idx >= 0
    assert (tab.idx[real] <= np.broadcast_to(i, tab.idx.shape)[real]).all()
    assert (tab.idx[real] >= np.broadcast_to(tab.start[:, None], tab.idx.shape)[real]).all()
    assert (tab.idx[:, -1] == np.arange(tab.n)).all()  # last history bucket is the current one


def test_target_matches_brute_force():
    tab = make_table(build_grid(synth(seed=3), 60), K=5, L=10)
    for i in range(tab.n):
        j = np.arange(i + 1, min(i + 6, tab.end[i]))
        full = i + 5 < tab.end[i]
        assert tab.valid[i] == full
        assert tab.fut[i] == (full and bool(tab.attack[j].any()))


def test_scaler_fit_on_train_rows_only():
    tab = make_table(build_grid(synth(), 60), K=5, L=10)
    train = np.arange(tab.n // 2)
    b = models.fit(tab, train, parts=("lr",))
    np.testing.assert_allclose(b["scaler"].mu, tab.X[train].mean(0), rtol=1e-6)
    poisoned = tab.X.copy()
    poisoned[tab.n // 2:] = 1e6
    np.testing.assert_allclose(models.Scaler(poisoned[train]).mu, b["scaler"].mu)


def test_shuffle_permutes_history_only():
    tab = make_table(build_grid(synth(), 60), K=5, L=10)
    s = models.shuffled_idx(tab.idx, 0)
    assert (s[:, -1] == tab.idx[:, -1]).all()
    assert (np.sort(s, 1) == np.sort(tab.idx, 1)).all()
    assert (s != tab.idx).any()


def test_blocked_split_has_no_overlap():
    """Train horizon ends before the boundary; test history starts after it (same rule as train_demo.py)."""
    tab = make_table(build_grid(synth(), 60), K=5, L=10)
    bnd = tab.start + np.floor(0.6 * (tab.end - tab.start)).astype(int)
    i = np.arange(tab.n)
    tr, te = i < bnd - tab.K, i >= bnd + tab.L
    assert (i[tr] + tab.K < bnd[tr]).all()
    real = tab.idx[te] >= 0
    assert (np.where(real, tab.idx[te], 10**9) >= np.broadcast_to(bnd[te][:, None], tab.idx[te].shape)).all()
    assert not (tr & te).any()


def test_no_identifier_features():
    banned = ("ip", "port_num", "time", "stamp", "host", "bucket", "day")
    assert not [f for f in FEATURES if any(k in f.split("_", 1)[1] for k in banned)]
