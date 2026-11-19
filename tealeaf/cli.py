"""Command line entry point: `tealeaf`."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import replace

from .client import ChatClient, ClientError
from .config import load_config
from .registry import SUITES, all_cases, select_cases
from .runner import Runner
from .storage import Store

C = {"pass": "\033[32m", "fail": "\033[31m", "error": "\033[35m", "observed": "\033[36m", "skipped": "\033[90m",
     "dim": "\033[90m", "bold": "\033[1m", "yellow": "\033[33m", "reset": "\033[0m"}
CLR = "\r\033[K" if sys.stdout.isatty() else ""
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C = {k: "" for k in C}


def _fmt_score(s):
    return "  -  " if s is None else f"{s:5.2f}"


def _apply_overrides(cfg, a):
    if getattr(a, "base_url", None):
        cfg.target.base_url = a.base_url
    if getattr(a, "model", None):
        cfg.target.model = a.model
    if getattr(a, "judge_url", None) or getattr(a, "judge_model", None):
        cfg.judge = cfg.judge or replace(cfg.target, temperature=0.0, extra={})
        if a.judge_url:
            cfg.judge.base_url = a.judge_url
        if a.judge_model:
            cfg.judge.model = a.judge_model
    for k in ("samples", "conversation_turns", "concurrency"):
        v = getattr(a, k, None)
        if v:
            setattr(cfg.run, k, v)
    if getattr(a, "no_annotate", False):
        cfg.run.annotate = False
    if getattr(a, "no_code", False):
        cfg.run.allow_code_execution = False
    if getattr(a, "opik", False):
        cfg.opik.enabled = True
    return cfg


def _opik_factory(cfg):
    if not cfg.opik.enabled:
        return None

    def make():
        try:
            from .opik_sink import OpikSink
            return OpikSink(cfg.opik)
        except Exception as e:  # noqa: BLE001
            print(f"[opik] disabled: {e}", file=sys.stderr)
            return None
    return make


class Printer:
    def __init__(self, total: int, verbose: bool):
        self.total, self.verbose, self.done = total, verbose, 0

    def __call__(self, type_: str, p: dict):
        if type_ == "test_started":
            if CLR:
                print(f"{CLR}{C['dim']}  … {p['case_id']}{C['reset']}", end="", flush=True)
        elif type_ == "message" and self.verbose and p.get("thread") != "judge":
            role = p["role"]
            text = p["content"].strip().replace("\n", "\n      ")
            col = C["yellow"] if role in ("user", "interlocutor", "A", "B") else C["dim"] if role == "system" else ""
            print(f"    {col}{role:>11}: {text[:1200]}{C['reset']}")
        elif type_ == "test_finished":
            self.done += 1
            st = p["status"]
            interest = f" ✦{p['interest']:.0f}" if p.get("interest") is not None else ""
            flags = f" {C['dim']}[{', '.join(p['flags'][:4])}]{C['reset']}" if p.get("flags") else ""
            err = f" {C['error']}{p['error']}{C['reset']}" if p.get("error") else ""
            print(f"{CLR}[{self.done:>3}/{self.total}] {p['case_id']:<44} {C.get(st, '')}{st.upper():<8}{C['reset']} "
                  f"{_fmt_score(p.get('score'))}{interest} {C['dim']}({p['duration']}s){C['reset']}{flags}{err}")
            if p.get("summary"):
                print(f"          {C['dim']}{p['summary'][:220]}{C['reset']}")


def print_summary(run: dict):
    s = run.get("summary") or {}
    print(f"\n{C['bold']}Run {run['id']}{C['reset']}  model={run['target_model']}  status={run['status']}")
    print(f"{'suite':<18}{'kind':<13}{'n':>4}{'pass':>6}{'fail':>6}{'err':>5}{'obs':>5}{'avg':>8}")
    for name, v in (s.get("suites") or {}).items():
        avg = "-" if v.get("avg_score") is None else f"{v['avg_score']:.2f}"
        print(f"{name:<18}{v['kind']:<13}{v['n']:>4}{v['pass']:>6}{v['fail']:>6}{v['error']:>5}{v['observed']:>5}"
              f"{avg:>8}")
    if s.get("avg_score") is not None:
        print(f"\noverall avg score: {s['avg_score']:.3f}", end="")
    if s.get("tokens_per_sec"):
        print(f"   decode speed: {s['tokens_per_sec']} tok/s", end="")
    print()
    if s.get("top_interest"):
        print(f"\n{C['bold']}Most interesting{C['reset']}")
        for t in s["top_interest"]:
            print(f"  ✦{t['interest']:.0f}  {t['case_id']:<36} {C['dim']}{(t.get('summary') or '')[:150]}{C['reset']}")


# --------------------------------------------------------------------------- commands
async def cmd_run(a):
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    store.mark_stale_runs()
    try:
        cases = select_cases(a.suite, a.case, a.kind, a.exclude)
    except ValueError as e:
        sys.exit(str(e))
    if not cases:
        sys.exit("no cases match the selection (see `tealeaf list`)")
    opik_make = _opik_factory(cfg)
    printer = Printer(len(cases), a.verbose)
    runner = Runner(cfg, store, opik=opik_make() if opik_make else None, listeners=[printer])

    server_task = None
    if a.web:
        server_task = asyncio.create_task(_serve(cfg, store, a.host, a.port, opik_make))
        print(f"dashboard: http://{a.host}:{a.port}")

    try:
        run_id = await runner.prepare(cases, a.label or "")
    except ClientError as e:
        sys.exit(f"cannot reach target: {e}")
    jm = "(self)" if runner.judge_is_target else runner.judge_client.model
    print(f"{C['bold']}run {run_id}{C['reset']}  target={runner.target.model} @ {cfg.target.base_url}  judge={jm}  "
          f"cases={len(cases)}  db={cfg.db_path}")
    try:
        await runner.run(cases, a.label or "")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted")
    print_summary(store.get_run(run_id))
    if server_task:
        print(f"\nrun finished; dashboard still serving at http://{a.host}:{a.port} (Ctrl-C to quit)")
        await server_task


async def _serve(cfg, store, host, port, opik_make=None):
    import uvicorn

    from .web.app import create_app
    config = uvicorn.Config(create_app(cfg, store, opik_make), host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()


def cmd_serve(a):
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    print(f"dashboard: http://{a.host}:{a.port}   target={cfg.target.base_url}   db={cfg.db_path}")
    asyncio.run(_serve(cfg, store, a.host, a.port, _opik_factory(cfg)))


def cmd_list(a):
    all_cases()
    for s in SUITES.values():
        if a.kind and s.kind not in a.kind:
            continue
        print(f"\n{C['bold']}{s.name}{C['reset']} {C['dim']}({s.kind}, {len(s.cases)} cases) {s.description}{C['reset']}")
        if not a.brief:
            for c in s.cases:
                print(f"  {c.id:<44} {C['dim']}{c.title}{C['reset']}")


def cmd_runs(a):
    store = Store(load_config(a.config).db_path)
    store.mark_stale_runs()
    for r in store.list_runs(a.limit):
        s = r.get("summary") or {}
        avg = s.get("avg_score")
        print(f"{r['id']}  {r['status']:<11} {r['target_model'] or '?':<32} {r['done_cases']}/{r['total_cases']:<4} "
              f"avg={'-' if avg is None else f'{avg:.2f}'}  {r['label'] or ''}")


def cmd_show(a):
    store = Store(load_config(a.config).db_path)
    run = store.get_run(a.run_id) or next((r for r in store.list_runs(500) if r["id"].startswith(a.run_id)), None)
    if not run:
        sys.exit("run not found")
    if a.case:
        res = store.get_result(run["id"], a.case)
        if not res:
            sys.exit("case not found in run")
        print(json.dumps({k: res[k] for k in ("status", "score", "interest", "flags", "metrics", "notes", "judge",
                                              "error")}, indent=2, default=str))
        for m in store.get_messages(run["id"], a.case):
            print(f"\n[{m['thread']}] {C['bold']}{m['role']}{C['reset']}:")
            if m.get("reasoning"):
                print(f"{C['dim']}(reasoning) {m['reasoning'][:1500]}{C['reset']}")
            print(m["content"])
        return
    print_summary(run)
    for r in store.get_results(run["id"]):
        print(f"  {r['case_id']:<44} {C.get(r['status'], '')}{r['status']:<9}{C['reset']}{_fmt_score(r['score'])}"
              + (f" ✦{r['interest']:.0f}" if r["interest"] is not None else ""))


def cmd_compare(a):
    store = Store(load_config(a.config).db_path)
    ra = {r["case_id"]: r for r in store.get_results(a.run_a)}
    rb = {r["case_id"]: r for r in store.get_results(a.run_b)}
    print(f"{'case':<44}{'A':>7}{'B':>7}{'Δ':>7}")
    for cid in sorted(set(ra) | set(rb)):
        x, y = (ra.get(cid) or {}).get("score"), (rb.get(cid) or {}).get("score")
        d = "" if x is None or y is None else f"{y - x:+.2f}"
        print(f"{cid:<44}{_fmt_score(x):>7}{_fmt_score(y):>7}{d:>7}")


async def cmd_models(a):
    cfg = _apply_overrides(load_config(a.config), a)
    c = ChatClient(cfg.target)
    try:
        for m in await c.list_models():
            print(m)
    except ClientError as e:
        sys.exit(str(e))
    finally:
        await c.aclose()


async def cmd_chat(a):
    """Interactive chat with the target; saved as a run so it appears in the dashboard."""
    from .registry import Case
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    runner = Runner(cfg, store)
    case = Case(suite="manual", name="chat", title="Manual chat", fn=None, kind="probe",
                description="Interactive conversation from `tealeaf chat`.")
    try:
        await runner.prepare([case], a.label or "manual chat")
    except ClientError as e:
        sys.exit(f"cannot reach target: {e}")
    from .runner import Ctx
    ctx = Ctx(runner, case)
    store.update_result(runner.run_id, case.id, status="running", started_at=time.time())
    conv = ctx.conversation(a.system)
    print(f"chatting with {runner.target.model} (run {runner.run_id}). Empty line or Ctrl-D to finish.")

    try:
        while True:
            try:
                line = input(f"{C['yellow']}you>{C['reset']} ")
            except EOFError:
                break
            if not line.strip():
                break
            r = await conv.say(line)
            if r.reasoning and a.show_reasoning:
                print(f"{C['dim']}{r.reasoning}{C['reset']}")
            print(f"{C['bold']}model>{C['reset']} {r.content}\n{C['dim']}{r.stats()['tokens_per_sec']} tok/s{C['reset']}")
    finally:
        if conv.assistant_turns and cfg.run.annotate:
            print("annotating…")
            data = await ctx.judge.annotate(runner.transcript_text(case.id))
            if data:
                store.update_result(runner.run_id, case.id, judge=data, interest=data.get("interest"))
                print(f"✦{data.get('interest')} {data.get('summary')}")
        store.update_result(runner.run_id, case.id, status="observed", finished_at=time.time())
        store.update_run(runner.run_id, status="finished", finished_at=time.time(), done_cases=1,
                         summary=runner.summarize())
        runner.emit("run_finished", {"status": "finished"})
        await runner.target.aclose()


def cmd_mock(a):
    import uvicorn

    from .mock_server import create_mock_app
    print(f"mock OpenAI-compatible server: http://{a.host}:{a.port}/v1  (model: mock-1)")
    uvicorn.run(create_mock_app(a.delay), host=a.host, port=a.port, log_level="warning")


# --------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tealeaf", description="Tealeaf, a local LLM test harness: benchmarks + behavioral probes.")
    p.add_argument("--config", help="path to tealeaf.toml (default: ./tealeaf.toml, ~/.config/tealeaf/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def target_args(sp):
        sp.add_argument("--base-url", help="target OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
        sp.add_argument("-m", "--model", help="target model id (default: first listed by the server)")
        sp.add_argument("--judge-url", help="judge base URL (default: same as target)")
        sp.add_argument("--judge-model", help="judge model id (default: the target judges itself)")

    r = sub.add_parser("run", help="run suites against the target model")
    target_args(r)
    r.add_argument("-s", "--suite", action="append", help="suite name (repeatable). Default: all")
    r.add_argument("-c", "--case", action="append", help="case id glob, e.g. 'beliefs.*' (repeatable)")
    r.add_argument("-x", "--exclude", action="append", help="exclude case id glob (repeatable)")
    r.add_argument("-k", "--kind", action="append", choices=["traditional", "probe"], help="only this kind")
    r.add_argument("-l", "--label", help="label for this run")
    r.add_argument("--samples", type=int, help="samples for randomness/consistency probes")
    r.add_argument("--conversation-turns", type=int, help="turns per judge-driven conversation")
    r.add_argument("--concurrency", type=int)
    r.add_argument("--no-annotate", action="store_true", help="skip judge annotation of probes")
    r.add_argument("--no-code", action="store_true", help="don't execute model-written code")
    r.add_argument("--opik", action="store_true", help="also log traces to Opik")
    r.add_argument("-v", "--verbose", action="store_true", help="print transcripts as they happen")
    r.add_argument("--web", action="store_true", help="serve the dashboard while running")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8765)

    s = sub.add_parser("serve", help="serve the web dashboard (and start runs from it)")
    target_args(s)
    s.add_argument("--opik", action="store_true")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    li = sub.add_parser("list", help="list suites and cases")
    li.add_argument("-k", "--kind", action="append", choices=["traditional", "probe"])
    li.add_argument("--brief", action="store_true")

    rs = sub.add_parser("runs", help="list past runs")
    rs.add_argument("-n", "--limit", type=int, default=20)

    sh = sub.add_parser("show", help="show a run, or one case's transcript")
    sh.add_argument("run_id", help="run id (prefix ok)")
    sh.add_argument("case", nargs="?", help="case id, e.g. beliefs.identity")

    cp = sub.add_parser("compare", help="compare scores of two runs")
    cp.add_argument("run_a")
    cp.add_argument("run_b")

    mo = sub.add_parser("models", help="list models available at the target endpoint")
    target_args(mo)

    ch = sub.add_parser("chat", help="interactive conversation with the target (saved + annotated)")
    target_args(ch)
    ch.add_argument("--system", help="system prompt")
    ch.add_argument("-l", "--label")
    ch.add_argument("--show-reasoning", action="store_true")

    mk = sub.add_parser("mock-server", help="run a fake OpenAI-compatible model for development")
    mk.add_argument("--host", default="127.0.0.1")
    mk.add_argument("--port", type=int, default=11999)
    mk.add_argument("--delay", type=float, default=0.01, help="seconds between streamed tokens")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "run":
            asyncio.run(cmd_run(a))
        elif a.cmd == "serve":
            cmd_serve(a)
        elif a.cmd == "list":
            cmd_list(a)
        elif a.cmd == "runs":
            cmd_runs(a)
        elif a.cmd == "show":
            cmd_show(a)
        elif a.cmd == "compare":
            cmd_compare(a)
        elif a.cmd == "models":
            asyncio.run(cmd_models(a))
        elif a.cmd == "chat":
            asyncio.run(cmd_chat(a))
        elif a.cmd == "mock-server":
            cmd_mock(a)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
