"""
Tamper-evident SHA-256 alert ledger.

Each alert record is serialised canonically, concatenated with the previous
block's hash, and hashed. Changing any field of any record changes that block's
hash and therefore every block after it, so tampering is detectable by recomputing
the chain.

Why this is built from the analysis result rather than from UI session state:
a chain accumulated as the operator clicks around would depend on the order they
happened to click, so two people examining the same capture would produce
different hashes and neither could verify the other. Building it deterministically
from the windows, in time order, means the chain is a function of the evidence
alone -- reproducible by anyone with the same input, which is what makes it worth
having.

    chain  = build(windows, threshold)
    ok, at = verify(chain)
"""

import hashlib
import json

GENESIS = "0" * 64


def _digest(record: dict, prev_hash: str) -> str:
    """sort_keys makes the serialisation canonical, so the same record always
    produces the same digest regardless of dict insertion order."""
    payload = json.dumps(record, sort_keys=True, separators=(',', ':')) + prev_hash
    return hashlib.sha256(payload.encode()).hexdigest()


def build(windows, threshold=0.30, source=None, model=None):
    """One chained block per window whose damage_risk exceeds `threshold`.

    `windows` is infer.analyze()['windows']. Returns a list of blocks, each
    carrying its record, its prev_hash and its sha256.
    """
    chain, prev = [], GENESIS
    for w in windows:
        f = w['forecast']
        risk = float(f['damage_risk'])
        if risk <= threshold:
            continue
        sig = w['signal']
        record = {
            'seq': len(chain),
            'window_idx': int(w['idx']),
            'time_offset_s': round(float(w['time']), 3),
            'stage_id': int(w['stage']),
            'risk': round(risk, 4),
            'breach': round(float(w['breach']), 4),
            'signal': sig['name'],
            'evidence': sig['evidence'],
            'forecast': f['phrase'],
            'confidence': round(float(f['confidence']), 4),
        }
        if source is not None:
            record['source'] = source
        if model is not None:
            record['model'] = model
        h = _digest(record, prev)
        chain.append({'record': record, 'prev_hash': prev, 'sha256': h})
        prev = h
    return chain


def verify(chain):
    """Recompute every block. Returns (ok, first_bad_index_or_None)."""
    prev = GENESIS
    for i, block in enumerate(chain):
        if block['prev_hash'] != prev:
            return False, i
        if _digest(block['record'], prev) != block['sha256']:
            return False, i
        prev = block['sha256']
    return True, None


def head(chain):
    """The hash committing to the whole chain: publish this and every earlier
    block is fixed."""
    return chain[-1]['sha256'] if chain else GENESIS


def _check():
    import copy
    windows = [{
        'idx': i, 'time': i * 1.5, 'stage': 4, 'breach': 0.8,
        'signal': {'name': 'C2 beaconing', 'evidence': f'periodic callback {i}'},
        'forecast': {'damage_risk': 0.4 + 0.05 * i, 'phrase': 'Sustained C2',
                     'confidence': 0.6},
    } for i in range(6)]

    chain = build(windows, threshold=0.30)
    assert len(chain) == 6, len(chain)
    ok, bad = verify(chain)
    assert ok and bad is None, (ok, bad)

    # chaining is real: each block commits to the one before it
    for i in range(1, len(chain)):
        assert chain[i]['prev_hash'] == chain[i - 1]['sha256']
    assert chain[0]['prev_hash'] == GENESIS

    # tamper with block 2 -> block 2 fails and the head changes
    t = copy.deepcopy(chain)
    t[2]['record']['risk'] = 0.0001
    ok, bad = verify(t)
    assert not ok and bad == 2, (ok, bad)

    # rehashing block 2 alone does not repair it: block 3 still breaks
    t[2]['sha256'] = _digest(t[2]['record'], t[2]['prev_hash'])
    ok, bad = verify(t)
    assert not ok and bad == 3, f"chain should break at 3, got {bad}"

    # deleting a block is detected
    t2 = copy.deepcopy(chain)
    del t2[3]
    ok, bad = verify(t2)
    assert not ok, "deletion went undetected"

    # determinism: same input -> same head, independent of build order
    assert head(build(windows, 0.30)) == head(chain)
    # threshold filters
    assert len(build(windows, threshold=0.55)) == 2, len(build(windows, 0.55))
    assert head([]) == GENESIS

    print(f"ledger self-check OK: {len(chain)} blocks, head "
          f"{head(chain)[:16]}…, tamper + deletion + rehash all detected")


if __name__ == '__main__':
    _check()
