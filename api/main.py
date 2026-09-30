"""Netrikan API: serves the trained models and evaluation results to the React console (web/). Offline, no cloud calls.

Run from the repo root:  make api   (dev, with `make web` for the Vite dev server)   or   make demo   (built UI + API on :8000)
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, "src")

from . import corpus, dapt, response, z24  # noqa: E402

app = FastAPI(title="Netrikan", docs_url="/api/docs", openapi_url="/api/openapi.json")
for r in (dapt.router, z24.router, corpus.router, response.router):
    app.include_router(r)


def _warm():
    """Compute the default views once at startup so the first click in the UI is fast (the heavy work is cached)."""
    import logging
    log = logging.getLogger("uvicorn.error")
    jobs = [(f"DAPT2020 {d}", lambda d=d: dapt.run(d)) for d in dapt.mx()["dataset"]["days"]]
    jobs += [("ZeekData24 2024-03-03", lambda: z24.analyse("2024-03-03", 0.5)),
             ("CIC-IDS2017 Friday", lambda: corpus.prepare("cic17", "fri")),
             ("CTU-13", lambda: corpus.prepare("ctu13", "both"))]
    for name, job in jobs:
        try:
            job()
            log.info("warm: %s ready", name)
        except Exception as e:  # noqa: BLE001  (missing data/models: the endpoint reports it on request)
            log.warning("warm: %s skipped (%s)", name, e)


@app.on_event("startup")
def warm_caches():
    import threading
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True}


DIST = "web/dist"
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=f"{DIST}/assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = os.path.join(DIST, path)
        return FileResponse(f if path and os.path.isfile(f) else f"{DIST}/index.html")
