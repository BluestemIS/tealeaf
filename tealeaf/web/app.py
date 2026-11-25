"""FastAPI dashboard: live run view (SSE tailing the SQLite event log), history, highlights, beliefs."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..client import ClientError
from ..config import Config, derive_judge, restore_config
from ..discovery import DiscoveryError, discover, match_models, model_prompt
from ..registry import SUITES, all_cases, select_cases
from ..runner import Runner
from ..stability import compute, judge_consistency, new_group_id, resolve_runs
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
    repeat: int = 1  # >1 -> N runs per model, grouped for the stability view
    system_prompt: str | None = None  # run-level system prompt (None = config default)
    api_key: str | None = None  # target API key from the UI (blank/None = config/env); kept in memory, never stored


class ResumeRun(BaseModel):
    api_key: str | None = None  # keys aren't stored with a run; the UI resends the one it started with


class StabilityJudge(BaseModel):
    refs: list[str]  # a group id, or two or more run ids


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

    # The UI sends a typed-in key as a header rather than a query parameter, so it stays out of access logs.
    @app.get("/api/models")
    async def models(base_url: str | None = None, x_api_key: str | None = Header(None)):
        """Pull the model list from an endpoint (live), cache it, and attach test history."""
        url = (base_url or cfg.target.base_url).rstrip("/")
        ep = replace(cfg.target, base_url=url, api_key=x_api_key or cfg.target.api_key)
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

    @app.get("/api/models/prompt")
    async def prompt(model: str, base_url: str | None = None, x_api_key: str | None = Header(None)):
        """The model's chat template and built-in system prompt, when the server exposes them."""
        ep = replace(cfg.target, base_url=(base_url or cfg.target.base_url).rstrip("/"),
                     api_key=x_api_key or cfg.target.api_key)
        return (await model_prompt(ep, model)).to_dict()

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

    @app.post("/api/runs/{run_id}/resume")
    async def resume(run_id: str, errors: bool = False, req: ResumeRun | None = None):
        """Continue a stopped run in place with the settings it started with."""
        store.mark_stale_runs()
        r = store.get_run(run_id)
        if not r:
            raise HTTPException(404)
        if r["status"] in ("running", "cancelling", "queued"):
            raise HTTPException(409, f"run is {r['status']}")
        if (r.get("config") or {}).get("reannotated_from"):
            raise HTTPException(400, "re-annotation runs can't be resumed; annotate the source run again")
        rc = restore_config(cfg, r.get("config") or {})
        if req and req.api_key:
            rc = replace(rc, target=replace(rc.target, api_key=req.api_key))
        runner = Runner(rc, store,
                        opik=opik_factory() if opik_factory else None)
        redo = {"pending", "running", "skipped"} | ({"error"} if errors else set())
        if not any(x["status"] in redo for x in store.get_results(run_id)):
            raise HTTPException(400, "nothing left to run" + ("" if errors else " (errored cases need errors=true)"))
        try:
            await runner.target.resolve_model()
        except ClientError as e:
            raise HTTPException(502, str(e))
        tasks[run_id] = asyncio.create_task(runner.resume(run_id, include_errors=errors))
        tasks[run_id].add_done_callback(lambda _t, k=run_id: tasks.pop(k, None))
        return {"run_id": run_id}

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
        ui_key = (req.api_key or "").strip()
        if ui_key:
            rc.target.api_key = ui_key
        names = req.models or ([req.model] if req.model else [])
        if any(n.lower() == "all" or any(c in n for c in "*?[") for n in names):
            try:
                names = match_models(names, await discover(rc.target))
            except DiscoveryError as e:
                raise HTTPException(502, str(e))
        if not names:
            names = [""]  # empty -> runner picks the first chat model
        if req.judge_model or req.judge_base_url:
            derived = rc.judge is None
            rc.judge = replace(rc.judge or derive_judge(rc.target))
            if req.judge_model:
                rc.judge.model = req.judge_model
            if req.judge_base_url:
                rc.judge.base_url = req.judge_base_url.rstrip("/")
            # A judge derived from the target inherits its key; don't send a typed-in key to a different server.
            if derived and ui_key and rc.judge.api_key == ui_key and rc.judge.base_url != rc.target.base_url:
                rc.judge.api_key = derive_judge(replace(rc.target, api_key="")).api_key
        if req.samples:
            rc.run.samples = req.samples
        if req.conversation_turns:
            rc.run.conversation_turns = req.conversation_turns
        if req.system_prompt is not None:
            rc.run.system_prompt = req.system_prompt
        repeat = max(1, min(50, req.repeat or 1))
        plan = []
        for name in names:
            gid = new_group_id() if repeat > 1 else None
            plan += [(name, {"group": {"id": gid, "index": i + 1, "of": repeat}} if gid else None)
                     for i in range(repeat)]
        runners = []
        for i, (name, meta) in enumerate(plan):
            mc = replace(rc, target=replace(rc.target, model=name, extra=dict(rc.target.extra)))
            r = Runner(mc, store, opik=opik_factory() if opik_factory else None)
            try:
                await r.prepare(cases, req.label, status="running" if i == 0 else "queued", meta=meta)
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

    @app.get("/api/groups")
    def groups():
        return store.groups()

    def _resolve(refs: list[str]):
        try:
            return resolve_runs(store, [r for r in refs if r])
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/stability")
    def stability(refs: str):
        """refs: a group id, or comma-separated run ids."""
        key, ids = _resolve(refs.split(","))
        return compute(store, key, ids)

    @app.post("/api/stability/judge")
    async def stability_judge(req: StabilityJudge):
        key, ids = _resolve(req.refs)
        try:
            await judge_consistency(store, key, ids, cfg.judge_endpoint())
        except ClientError as e:
            raise HTTPException(502, str(e))
        return compute(store, key, ids)

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
