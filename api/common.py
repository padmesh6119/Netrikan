"""Shared helpers: JSON-safe responses (numpy scalars/arrays, NaN -> null) and timestamp conversion."""
from __future__ import annotations

import inspect
import json
import math
from functools import lru_cache, wraps

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import JSONResponse


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, pd.Timestamp):
        return ms(o)
    return o


class Clean(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(clean(content), separators=(",", ":")).encode()


def ms(t) -> int:
    """Naive wall-clock timestamp -> epoch milliseconds (the UI formats in UTC, so the clock time is preserved)."""
    return int(pd.Timestamp(t).value // 1_000_000)


def bucket_ms(bucket: np.ndarray, bucket_s: int = 60) -> np.ndarray:
    return bucket.astype("int64") * bucket_s * 1000


@lru_cache(maxsize=None)
def read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(503, f"{path} not found. Train the models first (see README).")


def shap_top(contrib: np.ndarray, names: list[str], n: int) -> list[dict]:
    o = np.argsort(-np.abs(contrib))[:n]
    return [{"feature": names[q], "value": float(contrib[q])} for q in o]


def out(fn):
    """Return numpy-heavy results as a pre-rendered Clean response (skips FastAPI's encoder, which rejects numpy)."""
    if inspect.iscoroutinefunction(fn):
        @wraps(fn)
        async def aw(*a, **k):
            return Clean(await fn(*a, **k))
        return aw

    @wraps(fn)
    def w(*a, **k):
        return Clean(fn(*a, **k))
    return w


def cached(maxsize: int):
    """lru_cache that computes each key once even under concurrent requests (per-key lock, single flight)."""
    import threading
    from collections import OrderedDict

    def deco(fn):
        store: OrderedDict = OrderedDict()
        locks: dict = {}
        guard = threading.Lock()

        @wraps(fn)
        def w(*args):
            with guard:
                if args in store:
                    store.move_to_end(args)
                    return store[args]
                lock = locks.setdefault(args, threading.Lock())
            with lock:
                with guard:
                    if args in store:
                        return store[args]
                value = fn(*args)
                with guard:
                    store[args] = value
                    while len(store) > maxsize:
                        store.popitem(last=False)
                    locks.pop(args, None)
                return value

        w.cache_clear = lambda: store.clear()
        return w
    return deco
