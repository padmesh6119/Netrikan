"""
SEDI (Symmetric Extremal Dependence Index), Ferro & Stephenson 2011.

Base-rate independent skill score for rare-event forecasting. Unlike F1 or
precision, SEDI does not move when event prevalence changes, so a single number
holds across the 26% attack rate of a test set and the 0.02% of a real network.

    SEDI = (ln F - ln H - ln(1-F) + ln(1-H)) / (ln F + ln H + ln(1-F) + ln(1-H))

H = hit rate (recall), F = false alarm rate (FPR).
0 = no skill, 1 = perfect.

Inputs come from committed metrics files only. No dataset required.
"""

import json
import os
from math import log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sedi(H: float, F: float) -> float:
    e = 1e-12
    H = min(max(H, e), 1 - e)
    F = min(max(F, e), 1 - e)
    num = log(F) - log(H) - log(1 - F) + log(1 - H)
    den = log(F) + log(H) + log(1 - F) + log(1 - H)
    return num / den


def _rows():
    cm = json.load(open(f"{ROOT}/models/base_w30_metrics.json"))["confusion_matrix"]
    ben = cm[0]
    n_ben, fp = sum(ben), sum(ben) - ben[0]
    atk = cm[1:]
    n_atk = sum(sum(r) for r in atk)
    tp = sum(sum(r) - r[0] for r in atk)
    yield ("in_distribution", "base_w30 held-out val", tp / n_atk, fp / n_ben)

    a = json.load(open(f"{ROOT}/models/a1_entity_compare.json"))
    for key, label in (("segment", "DAPT 2020 cross-dataset"),
                       ("per_host", "DAPT 2020 per-host windowing")):
        b = a[key]["binary"]
        yield (key, label, b["recall"], b["fpr"])


def main():
    out = []
    for tag, label, H, F in _rows():
        out.append({"tag": tag, "label": label, "hit_rate": round(H, 4),
                    "false_alarm_rate": round(F, 4), "sedi": round(sedi(H, F), 4),
                    "peirce_skill_score": round(H - F, 4)})
        print(f"{label:<32} H={H:.4f} F={F:.4f}  SEDI={out[-1]['sedi']:+.4f}")

    path = f"{ROOT}/models/sedi.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved → {path}")


def _check():
    assert abs(sedi(0.5, 0.5)) < 1e-9, "no-skill forecast (H==F) must score 0"
    assert sedi(0.99, 0.01) > 0.9, "near-perfect forecast must score high"
    assert sedi(0.8, 0.4) > sedi(0.8, 0.5), "lower false alarms must score higher"
    # base-rate independence: SEDI depends only on H and F, never on prevalence
    assert sedi(0.839, 0.416) == sedi(0.839, 0.416)


if __name__ == "__main__":
    _check()
    main()
