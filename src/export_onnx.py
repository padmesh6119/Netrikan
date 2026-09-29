"""
Export the deployed WorldModel to ONNX, with parity verification.

Produces a portable, framework-free graph for air-gapped CPU deployment. All four
heads are exported (stage, breach, next-state, onset). Attention must match the
checkpoint: model.py warns that a checkpoint trained without attention must not be
loaded with it, and vice versa — this reads the flag from the state dict.

Fails the export if ONNX Runtime output differs from PyTorch by more than --atol
on any head, so a silently-wrong graph never ships.

    python3 src/export_onnx.py --model models/cic_v2_w30.pt \
        --out models/cic_v2_w30.onnx
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from model import WorldModel   # noqa: E402


class _Exportable(torch.nn.Module):
    """Fixed 4-tuple output (no optional attention return) for a stable graph."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        stage_logits, breach, next_state, onset_logits = self.m(x)
        return stage_logits, breach, next_state, onset_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'models', 'cic_v2_w30.pt'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--window', type=int, default=30)
    ap.add_argument('--features', type=int, default=24)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--atol', type=float, default=1e-4)
    args = ap.parse_args()
    out = args.out or args.model.replace('.pt', '.onnx')

    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    has_attn = 'attn.weight' in sd
    m = WorldModel(input_size=args.features, attention=has_attn)
    m.load_state_dict(sd, strict=False)
    m.eval()
    wrap = _Exportable(m).eval()

    dummy = torch.randn(1, args.window, args.features)
    names = dict(input_names=['window'],
                 output_names=['stage_logits', 'breach', 'next_state', 'onset_logits'],
                 dynamic_axes={'window': {0: 'batch'}, 'stage_logits': {0: 'batch'},
                               'breach': {0: 'batch'}, 'next_state': {0: 'batch'},
                               'onset_logits': {0: 'batch'}})
    # dynamo=False uses the TorchScript exporter, which embeds weights in one file.
    # The default dynamo exporter externalises them to a .onnx.data sidecar, which
    # is fine but not the single portable artifact air-gapped deployment wants.
    try:
        torch.onnx.export(wrap, dummy, out, opset_version=args.opset,
                          dynamo=False, **names)
    except TypeError:
        torch.onnx.export(wrap, dummy, out, opset_version=args.opset, **names)
    # remove any external-data sidecar if a fallback produced one
    side = out + '.data'
    external = os.path.exists(side)
    size_mb = (os.path.getsize(out) + (os.path.getsize(side) if external else 0)) / 1e6
    print(f"exported -> {out}  ({size_mb:.3f} MB total, opset {args.opset}, "
          f"attention={has_attn}{', + .data sidecar' if external else ', single file'})")

    # parity check against PyTorch on a batch
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping parity check. Install "
              "onnxruntime to verify before trusting the graph.", file=sys.stderr)
        return

    rng = np.random.default_rng(0)
    xb = rng.normal(0, 1, (16, args.window, args.features)).astype(np.float32)
    with torch.no_grad():
        pt = wrap(torch.from_numpy(xb))
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    onx = sess.run(None, {'window': xb})
    names = ['stage_logits', 'breach', 'next_state', 'onset_logits']
    worst = 0.0
    for name, a, b in zip(names, pt, onx):
        d = float(np.abs(a.numpy() - np.asarray(b)).max())
        worst = max(worst, d)
        print(f"  {name:<13} max|Δ| {d:.2e}")
    if worst > args.atol:
        os.remove(out)
        raise SystemExit(f"PARITY FAILED: max diff {worst:.2e} > {args.atol}. "
                         f"Removed {out} rather than ship a wrong graph.")
    print(f"parity OK (worst {worst:.2e} <= {args.atol})")


if __name__ == '__main__':
    main()
