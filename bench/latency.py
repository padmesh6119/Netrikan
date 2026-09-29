"""
Inference latency and footprint, PyTorch vs ONNX, single CPU thread.

Air-gapped CPU deployment on substation/telecom hardware is a stated design goal,
and we had no latency number. Reports median / p95 / p99 per-window latency,
throughput (windows/sec), and model file size, for both paths. Hardware is
recorded in the output.

    python3 bench/latency.py --model models/cic_v2_w30.pt \
        --onnx models/cic_v2_w30.onnx --out models/latency.json
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from model import WorldModel   # noqa: E402


def _pct(a):
    return {"median_ms": round(float(np.percentile(a, 50)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "mean_ms": round(float(np.mean(a)), 4),
            "throughput_per_s": round(1000.0 / float(np.mean(a)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'cic_v2_w30.pt'))
    ap.add_argument('--onnx', default=None)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--features', type=int, default=24)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'latency.json'))
    args = ap.parse_args()

    # single thread, to represent a constrained edge deployment honestly
    torch.set_num_threads(1)

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    m = WorldModel(input_size=args.features, attention='attn.weight' in sd)
    m.load_state_dict(sd, strict=False)
    m.eval()

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(0, 1, (1, args.window, args.features)).astype(np.float32))

    out = {"hardware": {"platform": platform.platform(),
                        "processor": platform.processor() or platform.machine(),
                        "torch_threads": 1},
           "model_file": os.path.basename(args.model),
           "model_size_mb": round(os.path.getsize(args.model) / 1e6, 3)}

    # PyTorch
    with torch.no_grad():
        for _ in range(args.warmup):
            m(x)
        t = []
        for _ in range(args.iters):
            s = time.perf_counter()
            m(x)
            t.append((time.perf_counter() - s) * 1000.0)
    out["pytorch"] = _pct(np.array(t))
    print(f"PyTorch: median {out['pytorch']['median_ms']} ms, "
          f"{out['pytorch']['throughput_per_s']} windows/s (1 thread)")

    # ONNX
    onnx_path = args.onnx or args.model.replace('.pt', '.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(onnx_path, sess_options=so,
                                        providers=['CPUExecutionProvider'])
            xb = x.numpy()
            for _ in range(args.warmup):
                sess.run(None, {'window': xb})
            t = []
            for _ in range(args.iters):
                s = time.perf_counter()
                sess.run(None, {'window': xb})
                t.append((time.perf_counter() - s) * 1000.0)
            out["onnx"] = {**_pct(np.array(t)),
                           "model_size_mb": round(os.path.getsize(onnx_path) / 1e6, 3)}
            print(f"ONNX:    median {out['onnx']['median_ms']} ms, "
                  f"{out['onnx']['throughput_per_s']} windows/s (1 thread)")
        except ImportError:
            out["onnx"] = {"note": "onnxruntime not installed"}
            print("onnxruntime not installed — PyTorch path only", file=sys.stderr)
    else:
        out["onnx"] = {"note": f"{onnx_path} not found; run src/export_onnx.py first"}

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
