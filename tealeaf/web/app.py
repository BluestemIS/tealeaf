"""FastAPI dashboard: live run view (SSE tailing the SQLite event log), history, highlights, beliefs."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..client import ClientError
from ..config import Config
from ..discovery import DiscoveryError, discover, match_models
from ..registry import SUITES, all_cases, select_cases
from ..runner import Runner
from ..storage import Store

STATIC = Path(__file__).parent / "static"


class StartRun(BaseModel):
    suites: list[str] = []
    cases: list[str] = []
    kinds: list[str] = []
    label: str = ""
    model: str | None = None
    models: list[str] = []  # several models -> one queued run per model, executed in order
    base_url: str | None = None
    judge_model: str | None = None
    judge_base_url: str | None = None
    samples: int | None = None
    conversation_turns: int | None = None


def create_app(cfg: Config, store: Store, opik_factory=None) -> FastAPI:
    app = FastAPI(title="Tealeaf")
    tasks: dict[str, asyncio.Task] = {}
    store.mark_stale_runs()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/tealeaf.svg")
    def logo():
        return FileResponse(STATIC / "tealeaf.svg", media_type="image/svg+xml")

    @app.get("/api/config")
    def config():
        return cfg.public()

    @app.get("/api/suites")
    def suites():
        all_cases()
        return [s.info() for s in SUITES.values()]

    @app.get("/api/models")
    async def models(base_url: str | None = None, api_key: str | None = None):
        """Pull the model list from an endpoint (live), cache it, and attach test history."""
        url = (base_url or cfg.target.base_url).rstrip("/")
        ep = replace(cfg.target, base_url=url, api_key=api_key if api_key is not None else cfg.target.api_key)
        history = store.model_history()
        try:
            found = [m.to_dict() for m in await discover(ep)]
            store.save_models(url, found)
            error = None
        except DiscoveryError as e:
            found = [{**r["info"], "stale": True, "last_seen": r["last_seen"]} for r in store.known_models(url)
                     if isinstance(r.get("info"), dict)]
            error = str(e)
        for m in found:
            h = history.get((url, m["id"])) or {}
            m["runs"] = h.get("runs", 0)
            m["latest"] = h.get("latest")
        return {"base_url": url, "models": found, "error": error, "configured": cfg.target.model}

    @app.get("/api/runs")
    def runs(limit: int = 100):
        store.mark_stale_runs()
        return store.list_runs(limit)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        r = store.get_run(run_id)
        if not r:
            raise HTTPException(404)
        return {"run": r, "results": store.get_results(run_id)}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        r = store.get_run(run_id)
        if r and r["status"] in ("running", "cancelling", "queued"):
            raise HTTPException(409, "run is active; cancel it first")
        store.delete_run(run_id)
        return {"ok": True}

    @app.get("/api/runs/{run_id}/cases/{case_id}")
    def case(run_id: str, case_id: str):
        res = store.get_result(run_id, case_id)
        if not res:
            raise HTTPException(404)
        info = next((c.info() for c in all_cases() if c.id == case_id), None)
        return {"result": res, "case": info, "messages": store.get_messages(run_id, case_id)}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        store.request_cancel(run_id)
        return {"ok": True}

    @app.post("/api/runs")
    async def start(req: StartRun):
        try:
            cases = select_cases(req.suites or None, req.cases or None, req.kinds or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not cases:
            raise HTTPException(400, "no cases selected")
        rc = replace(cfg, target=replace(cfg.target, extra=dict(cfg.target.extra)), run=replace(cfg.run))
        if req.base_url:
            rc.target.base_url = req.base_url.rstrip("/")
        names = req.models or ([req.model] if req.model else [])
        if any(n.lower() == "all" or any(c in n for c in "*?[") for n in names):
            try:
                names = match_models(names, await discover(rc.target))
            except DiscoveryError as e:
                raise HTTPException(502, str(e))
        if not names:
            names = [""]  # empty -> runner picks the first chat model
        if req.judge_model or req.judge_base_url:
            rc.judge = replace(rc.judge or replace(rc.target, temperature=0.0, extra={}))
            if req.judge_model:
                rc.judge.model = req.judge_model
            if req.judge_base_url:
                rc.judge.base_url = req.judge_base_url
        if req.samples:
            rc.run.samples = req.samples
        if req.conversation_turns:
            rc.run.conversation_turns = req.conversation_turns
        runners = []
        for i, name in enumerate(names):
            mc = replace(rc, target=replace(rc.target, model=name, extra=dict(rc.target.extra)))
            r = Runner(mc, store, opik=opik_factory() if opik_factory else None)
            try:
                await r.prepare(cases, req.label, status="running" if i == 0 else "queued")
            except ClientError as e:
                for prev in runners:
                    store.request_cancel(prev.run_id)
                    store.update_run(prev.run_id, status="cancelled")
                raise HTTPException(502, str(e))
            runners.append(r)

        async def sequence():
            for r in runners:
                await r.run(cases, req.label)

        key = runners[0].run_id
        tasks[key] = asyncio.create_task(sequence())
        tasks[key].add_done_callback(lambda _t, k=key: tasks.pop(k, None))
        return {"run_id": key, "run_ids": [r.run_id for r in runners], "models": [r.target.model for r in runners],
                "cases": len(cases)}

    @app.get("/api/highlights")
    def highlights(min_interest: float = 5):
        return store.highlights(min_interest)

    @app.get("/api/beliefs")
    def beliefs():
        out = []
        for r in store.annotated():
            j = r.get("judge")
            if isinstance(j, dict):
                for b in j.get("beliefs") or []:
                    if isinstance(b, dict) and b.get("topic"):
                        out.append({"model": r["target_model"], "run_id": r["run_id"], "case_id": r["case_id"],
                                    "topic": str(b.get("topic")), "stance": str(b.get("stance", "")),
                                    "ts": r["started_at"]})
        return out

    @app.get("/api/compare")
    def compare(a: str, b: str):
        ra = {r["case_id"]: r for r in store.get_results(a)}
        rb = {r["case_id"]: r for r in store.get_results(b)}
        rows = []
        for cid in sorted(set(ra) | set(rb)):
            x, y = ra.get(cid, {}), rb.get(cid, {})
            rows.append({"case_id": cid, "title": x.get("title") or y.get("title"), "suite": x.get("suite") or
                         y.get("suite"), "a": {k: x.get(k) for k in ("status", "score", "interest")},
                         "b": {k: y.get(k) for k in ("status", "score", "interest")}})
        return {"a": store.get_run(a), "b": store.get_run(b), "rows": rows}

    @app.get("/api/events")
    async def events(request: Request, after: int | None = None):
        last = request.headers.get("last-event-id")
        cursor = int(last) if last and last.isdigit() else (after if after is not None else store.last_event_id())

        async def gen():
            nonlocal cursor
            yield "retry: 2000\n\n"
            idle = 0
            while True:
                if await request.is_disconnected():
                    break
                evs = store.events_after(cursor)
                for e in evs:
                    cursor = e["id"]
                    data = json.dumps({"type": e["type"], "run_id": e["run_id"], **(e["payload"] or {})},
                                      default=str)
                    yield f"id: {e['id']}\nevent: {e['type']}\ndata: {data}\n\n"
                idle = 0 if evs else idle + 1
                if idle and idle % 30 == 0:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.25 if evs else 0.5)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
