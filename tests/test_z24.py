import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from netrikan import campaign as C  # noqa: E402
from netrikan.zeek import TECH_LABEL, TECHS, _to_flows  # noqa: E402

NT, K, WIN = C.NT, C.K, C.WIN


def seqs(lengths=(400, 350), seed=0, p=0.03):
    rng = np.random.default_rng(seed)
    parts, n = {k: [] for k in ("host", "wk", "bucket", "X", "vol", "F", "Fp", "start", "end")}, 0
    for i, T in enumerate(lengths):
        f = rng.random((T, NT)) < p
        parts["host"].append(np.full(T, f"h{i}", dtype=object)); parts["wk"].append(np.full(T, "w", dtype=object))
        parts["bucket"].append(np.arange(T) + 1000 * i); parts["X"].append(rng.normal(size=(T, 38)).astype(np.float32))
        parts["vol"].append(rng.random((T, 2)).astype(np.float32)); parts["F"].append(f); parts["Fp"].append(f.copy())
        parts["start"].append(np.full(T, n)); parts["end"].append(np.full(T, n + T))
        n += T
    return C.Seqs(**{k: np.concatenate(v) for k, v in parts.items()})


def test_zeek_history_flag_counts():
    df = pd.DataFrame({"wk": ["w"], "uid": ["u"], "ts_epoch": [1.7e9], "Src IP": ["143.88.1.1"], "Dst IP": ["143.88.1.2"], "dport": [22], "proto": ["tcp"],
                       "dur": [1.5], "op": [4], "rp": [3], "ob": [10], "rb": [20], "oib": [100], "rib": [200], "history": ["ShADadFf"]})
    for t in TECHS:
        df[f"t_{t}"] = t == "T1110"
    f = _to_flows(df).iloc[0]
    assert (f["SYN Flag Count"], f["ACK Flag Count"], f["FIN Flag Count"], f["RST Flag Count"], f["PSH Flag Count"]) == (2, 2, 2, 0, 2)
    assert f["Flow Duration"] == 1.5e6 and f["Packet Length Mean"] == 300 / 7 and f["attack"] and f["t_T1110"]


def test_history_features_are_causal():
    """Changing flags after minute t must not change the features at or before t."""
    S = seqs()
    t = 200
    a = C.sched_feats(S, S.F)
    F2 = S.F.copy()
    F2[t + 1:S.end[0]] = True
    b = C.sched_feats(S, F2)
    np.testing.assert_array_equal(a[:t + 1], b[:t + 1])
    assert (a[t + 1:S.end[0]] != b[t + 1:S.end[0]]).any()


def test_features_do_not_cross_sequences():
    S = seqs()
    F2 = S.F.copy()
    F2[:S.end[0]] = True  # flip the first sequence
    a, b = C.sched_feats(S, S.F), C.sched_feats(S, F2)
    np.testing.assert_array_equal(a[S.end[0]:], b[S.end[0]:])


def test_target_matches_brute_force():
    S = seqs(seed=2, p=0.05)
    fut, valid = C.targets(S)
    for i in range(S.n):
        full = i + K < S.end[i]
        assert valid[i] == (i + K <= S.end[i] - 1)
        for j in range(NT):
            want = full and bool(S.F[i + 1:i + K + 1, j].any())
            assert fut[i, j] == want


def test_feature_values_match_definition():
    S = seqs(seed=3)
    f = C.sched_feats(S, S.F)
    i = 250
    for j in range(NT):
        col = S.F[S.start[i]:i + 1, j]
        idx = np.flatnonzero(col)
        since = (len(col) - 1 - idx[-1]) if len(idx) and len(col) - 1 - idx[-1] < WIN else WIN
        assert f[i, j] == min(since, WIN)
        assert f[i, NT + j] == col[-60:].sum()
        assert f[i, 2 * NT + j] == col[-WIN:].sum()


def test_shuffle_preserves_counts_but_not_recency():
    S = seqs(seed=4, p=0.05)
    a, b = C.sched_feats(S, S.F), C.sched_feats(S, S.F, shuffle=True)
    np.testing.assert_array_equal(a[:, 2 * NT:], b[:, 2 * NT:])   # n150 is order-invariant
    assert (a[:, :NT] != b[:, :NT]).mean() > 0.3                  # 'since' is not
    assert (a[:, NT:2 * NT] != b[:, NT:2 * NT]).mean() > 0.1      # n60 is not


def test_hazard_uses_only_time_since_last_burst():
    since = np.r_[np.full(100, 10.0), np.full(100, 100.0)]
    y = np.r_[np.zeros(100), np.ones(100)]
    h = C.Hazard().fit(since, y)
    assert h.predict(np.array([10.0]))[0] < 0.2 < 0.8 < h.predict(np.array([100.0]))[0]


def test_technique_labels_are_five_booleans():
    assert TECH_LABEL == [f"t_{t}" for t in TECHS] and len(TECHS) == 5
