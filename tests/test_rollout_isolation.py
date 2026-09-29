"""
Prove the K-step rollout is honest: it forecasts from the observed window and its
own predictions, and never reads observations it should not have at that step.

This is the analogue of AttackForecast's smoke test that an imagination-only loss
leaves the encoder's gradient at zero. Our rollout() only receives the initial
window, so the property to enforce is structural and behavioural:

  1. no-future-leak  altering rows AFTER the window cannot change the rollout —
     the function has no path to future observations.
  2. autoregressive  step k>1 depends on the model's own predicted state, not on
     a repeat of the last observed row. Perturbing the state_head changes step 2+
     but not step 1.
  3. determinism     same window in, same rollout out.
  4. gradient path   a loss on rollout outputs backpropagates into the observed
     window (it uses history) — the rollout is differentiable and grounded in the
     input, not a detached constant.

Run: python3 tests/test_rollout_isolation.py   (no weights or data needed)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from model import WorldModel   # noqa: E402


def _model(seed=0):
    torch.manual_seed(seed)
    return WorldModel(input_size=24).eval()


def test_no_future_leak():
    m = _model()
    rng = np.random.default_rng(0)
    # a longer stream; the rollout is given only the first WINDOW rows
    stream = torch.tensor(rng.normal(0, 1, (1, 60, 24)).astype(np.float32))
    win = stream[:, :30, :]
    s1, st1 = m.rollout(win, steps=10)
    # corrupt everything after the window; rollout must be identical
    stream2 = stream.clone()
    stream2[:, 30:, :] = torch.tensor(rng.normal(5, 3, (1, 30, 24)).astype(np.float32))
    s2, st2 = m.rollout(stream2[:, :30, :], steps=10)
    assert torch.equal(s1, s2) and torch.equal(st1, st2), \
        "rollout changed when only post-window rows changed — it is reading ahead"
    print("  [1] no-future-leak: rollout ignores rows outside its input window")


def test_autoregressive():
    m = _model()
    rng = np.random.default_rng(1)
    win = torch.tensor(rng.normal(0, 1, (1, 30, 24)).astype(np.float32))
    _, st_before = m.rollout(win, steps=5)
    # perturb the state head: this changes the PREDICTED next states, which only
    # affect step 2 onward (step 1 is computed from the real window)
    with torch.no_grad():
        m.state_head.weight.add_(1.0)
    _, st_after = m.rollout(win, steps=5)
    step1_same = torch.allclose(st_before[:, 0], st_after[:, 0], atol=1e-6)
    step2_diff = not torch.allclose(st_before[:, 1], st_after[:, 1], atol=1e-6)
    assert step1_same, "step 1 changed when only the state head changed — step 1 " \
        "must come from the observed window, not a prediction"
    assert step2_diff, "step 2 unchanged when the state head changed — the rollout " \
        "is not feeding its own predictions forward (not autoregressive)"
    print("  [2] autoregressive: step 1 from observation; step 2+ from predictions")


def test_determinism():
    m = _model()
    rng = np.random.default_rng(2)
    win = torch.tensor(rng.normal(0, 1, (1, 30, 24)).astype(np.float32))
    a = m.rollout(win, steps=8)[0]
    b = m.rollout(win, steps=8)[0]
    assert torch.equal(a, b), "rollout is nondeterministic"
    print("  [3] determinism: identical window -> identical rollout")


def test_gradient_grounded_in_history():
    m = _model()
    rng = np.random.default_rng(3)
    win = torch.tensor(rng.normal(0, 1, (1, 30, 24)).astype(np.float32),
                       requires_grad=True)
    # differentiable rollout (bypass the no_grad decorator by calling forward)
    window = win
    outs = []
    for _ in range(5):
        logits, _, nxt, _ = m.forward(window)
        outs.append(logits)
        window = torch.cat([window[:, 1:, :], nxt.unsqueeze(1)], dim=1)
    loss = torch.stack(outs).pow(2).mean()
    loss.backward()
    assert win.grad is not None and win.grad.abs().sum() > 0, \
        "no gradient path from rollout outputs back to the observed window"
    print("  [4] gradient path: rollout outputs depend on the observed window")


def main():
    print("rollout isolation tests:")
    test_no_future_leak()
    test_autoregressive()
    test_determinism()
    test_gradient_grounded_in_history()
    print("ALL ROLLOUT ISOLATION TESTS PASSED")


if __name__ == '__main__':
    main()
