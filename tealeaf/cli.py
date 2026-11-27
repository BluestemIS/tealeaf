"""Command line entry point: `tealeaf`."""
from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path

from .client import ClientError
from .config import derive_judge, key_for, load_config, restore_config
from .discovery import DiscoveryError, ModelInfo, PromptInfo, discover, match_models, model_prompt
from .registry import SUITES, all_cases, select_cases
from .runner import Runner
from .stability import compute, judge_consistency, new_group_id, resolve_runs
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
    derived = False
    if getattr(a, "judge_url", None) or getattr(a, "judge_model", None):
        derived = cfg.judge is None
        if derived:
            cfg.judge = derive_judge(cfg.target, a.judge_url)
        elif a.judge_url:
            # A configured judge's key belongs to its configured server; a different URL gets TEALEAF_JUDGE_API_KEY.
            cfg.judge.api_key = key_for(a.judge_url, cfg.judge) or os.environ.get("TEALEAF_JUDGE_API_KEY", "")
            cfg.judge.base_url = a.judge_url
        if a.judge_model:
            cfg.judge.model = a.judge_model
    if getattr(a, "api_key_env", None):
        key = _key_from_env(a.api_key_env)
        cfg.target.api_key = key
        # A judge derived from the target shares its key only when it's the same server.
        if derived and cfg.judge.base_url.rstrip("/") == cfg.target.base_url.rstrip("/"):
            cfg.judge.api_key = key
    for k in ("samples", "conversation_turns", "concurrency"):
        v = getattr(a, k, None)
        if v:
            setattr(cfg.run, k, v)
    if getattr(a, "run_system_file", None):
        cfg.run.system_prompt = Path(a.run_system_file).expanduser().read_text()
    if getattr(a, "run_system", None) is not None:
        cfg.run.system_prompt = a.run_system
    if getattr(a, "contamination_file", None):
        cfg.run.contamination_files = list(cfg.run.contamination_files) + a.contamination_file
    if getattr(a, "no_annotate", False):
        cfg.run.annotate = False
    if getattr(a, "no_code", False):
        cfg.run.allow_code_execution = False
    if getattr(a, "opik", False):
        cfg.opik.enabled = True
    return cfg


def _key_from_env(var: str) -> str:
    key = os.environ.get(var, "")
    if not key:
        sys.exit(f"--api-key-env: environment variable {var} is not set")
    return key


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
    cal = s.get("calibration")
    if cal:
        lean = "overconfident" if cal["overconfidence"] > 0.05 else "underconfident" if cal["overconfidence"] < -0.05 \
            else "well calibrated"
        print(f"calibration ({cal['n']} answers, from token probabilities): {cal['mean_confidence']:.0%} confident, "
              f"{cal['accuracy']:.0%} right, error {cal['ece']:.2f} ({lean})")
    if s.get("top_interest"):
        print(f"\n{C['bold']}Most interesting{C['reset']}")
        for t in s["top_interest"]:
            print(f"  ✦{t['interest']:.0f}  {t['case_id']:<36} {C['dim']}{(t.get('summary') or '')[:150]}{C['reset']}")


def print_model_table(runs: list[dict]):
    suites = list(dict.fromkeys(s for r in runs for s in ((r.get("summary") or {}).get("suites") or {})))
    width = max(len(r["target_model"]) for r in runs)
    print(f"\n{C['bold']}Model comparison{C['reset']} {C['dim']}(avg score per suite){C['reset']}")
    print(f"{'model':<{width}}  {'overall':>7}  " + "  ".join(f"{s[:10]:>10}" for s in suites))
    for r in runs:
        s = r.get("summary") or {}
        cells = []
        for name in suites:
            v = ((s.get("suites") or {}).get(name) or {}).get("avg_score")
            cells.append(f"{'-' if v is None else f'{v:.2f}':>10}")
        ov = s.get("avg_score")
        print(f"{r['target_model']:<{width}}  {'-' if ov is None else f'{ov:.2f}':>7}  " + "  ".join(cells))


# --------------------------------------------------------------------------- commands
async def cmd_run(a):
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    store.mark_stale_runs()
    store.prune_events()
    try:
        cases = select_cases(a.suite, a.case, a.kind, a.exclude)
    except ValueError as e:
        sys.exit(str(e))
    if not cases:
        sys.exit("no cases match the selection (see `tealeaf list`)")
    opik_make = _opik_factory(cfg)
    models = await _choose_models(cfg, store, interactive=sys.stdin.isatty() and not a.yes)
    if not models:
        sys.exit("no models selected")

    server_task = None
    if a.web:
        host, port, url = _listen(cfg, a)
        server_task = asyncio.create_task(_serve(cfg, store, host, port, opik_make))
        print(f"dashboard: {url}")

    # Create every run up front (queued) so the dashboard shows the whole plan, then run them in turn.
    # --repeat N: N runs per model, tied together by a group id for `tealeaf stability`.
    repeat = max(1, a.repeat or 1)
    runners, groups = [], []
    for m in models:
        gid = new_group_id() if repeat > 1 else None
        if gid:
            groups.append(gid)
        for i in range(repeat):
            mcfg = replace(cfg, target=replace(cfg.target, model=m, extra=dict(cfg.target.extra)))
            printer = Printer(len(cases), a.verbose)
            r = Runner(mcfg, store, opik=opik_make() if opik_make else None, listeners=[printer])
            meta = {"group": {"id": gid, "index": i + 1, "of": repeat}} if gid else None
            try:
                await r.prepare(cases, a.label or "", status="queued" if runners else "running", meta=meta)
            except ClientError as e:
                sys.exit(f"cannot reach target: {e}")
            runners.append(r)

    finished = []
    for i, runner in enumerate(runners):
        jm = "(self)" if runner.judge_is_target else runner.judge_client.model
        head = f"[run {i + 1}/{len(runners)}] " if len(runners) > 1 else ""
        print(f"\n{C['bold']}{head}run {runner.run_id}{C['reset']}  target={runner.target.model} @ "
              f"{cfg.target.base_url}  judge={jm}  cases={len(cases)}  db={cfg.db_path}")
        try:
            await runner.run(cases, a.label or "")
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\ninterrupted")
            for rest in runners[i + 1:]:
                store.request_cancel(rest.run_id)
            print_summary(store.get_run(runner.run_id))
            break
        except Exception as e:  # noqa: BLE001 - the run is already marked "error"; go on to the next one
            print(f"\n{C['error']}run {runner.run_id} crashed: {type(e).__name__}: {e}{C['reset']}", file=sys.stderr)
        finished.append(store.get_run(runner.run_id))
        print_summary(finished[-1])
    if len(finished) > 1 and len(models) > 1:
        print_model_table(finished)
    for gid in groups:
        ids = store.group_run_ids(gid)
        print_stability(compute(store, gid, ids), brief=True)
        print(f"{C['dim']}details: tealeaf stability {gid}   (add --judge to compare beliefs across runs){C['reset']}")
    if server_task:
        print(f"\nrun finished; dashboard still serving at {url} (Ctrl-C to quit)")
        await server_task


def _listen(cfg, a) -> tuple[str, int, str]:
    """Host/port from --host/--port, else [web] in tealeaf.toml. Returns (host, port, url to show)."""
    host = a.host or cfg.web.host
    port = a.port or cfg.web.port
    if host in ("0.0.0.0", "::", ""):
        url = f"http://{socket.gethostname()}:{port}  (listening on all interfaces; no authentication)"
    else:
        url = f"http://{'[' + host + ']' if ':' in host else host}:{port}"
    return host, port, url


async def _serve(cfg, store, host, port, opik_make=None):
    import uvicorn

    from .web.app import create_app
    config = uvicorn.Config(create_app(cfg, store, opik_make), host=host, port=port, log_level="warning",
                            timeout_graceful_shutdown=2)  # open SSE streams would otherwise block exit
    await uvicorn.Server(config).serve()


def cmd_serve(a):
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    host, port, url = _listen(cfg, a)
    print(f"dashboard: {url}   target={cfg.target.base_url}   db={cfg.db_path}")
    asyncio.run(_serve(cfg, store, host, port, _opik_factory(cfg)))


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


def _find_run(store: Store, run_id: str) -> dict | None:
    """A run by exact id or unique-enough prefix (the most recent match wins)."""
    hit = store.get_run(run_id) or next((r for r in store.list_runs(500) if r["id"].startswith(run_id)), None)
    return store.get_run(hit["id"]) if hit else None


def cmd_show(a):
    store = Store(load_config(a.config).db_path)
    run = _find_run(store, a.run_id)
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
            lps = (m.get("meta") or {}).get("logprob_stats")
            if a.tokens and m.get("logprobs"):
                print(_token_text(m["logprobs"]))
            else:
                print(m["content"])
            if lps:
                print(f"{C['dim']}perplexity {lps['perplexity']} · {lps['forks']} fork(s) · lowest token p={lps['min_p']}"
                      f"{'' if a.tokens else '  (--tokens shows each token)'}{C['reset']}")
        return
    print_summary(run)
    for r in store.get_results(run["id"]):
        print(f"  {r['case_id']:<44} {C.get(r['status'], '')}{r['status']:<9}{C['reset']}{_fmt_score(r['score'])}"
              + (f" ✦{r['interest']:.0f}" if r["interest"] is not None else ""))


def _token_text(tokens: list) -> str:
    """The answer with each uncertain token marked: word⟨0.42: alt 0.31⟩."""
    import math
    from .logprobs import visible
    out = []
    for tok, lp, alts, _ in visible(tokens):
        p = math.exp(lp)
        if p >= 0.6:
            out.append(tok)
            continue
        best = next((a for a in sorted(alts, key=lambda a: -a[1]) if a[0].strip() != tok.strip()), None)
        alt = f": {best[0].strip() or repr(best[0])} {math.exp(best[1]):.2f}" if best else ""
        col = C["fail"] if p < 0.3 else C["yellow"]
        out.append(f"{col}{tok}⟨{p:.2f}{alt}⟩{C['reset']}")
    return "".join(out)


def cmd_compare(a):
    store = Store(load_config(a.config).db_path)
    ra = {r["case_id"]: r for r in store.get_results(a.run_a)}
    rb = {r["case_id"]: r for r in store.get_results(a.run_b)}
    print(f"{'case':<44}{'A':>7}{'B':>7}{'Δ':>7}{'✦A':>6}{'✦B':>5}")
    for cid in sorted(set(ra) | set(rb)):
        x, y = (ra.get(cid) or {}).get("score"), (rb.get(cid) or {}).get("score")
        d = "" if x is None or y is None else f"{y - x:+.2f}"
        ia, ib = ((r.get(cid) or {}).get("interest") for r in (ra, rb))
        ia, ib = ("-" if v is None else f"{v:.0f}" for v in (ia, ib))
        print(f"{cid:<44}{_fmt_score(x):>7}{_fmt_score(y):>7}{d:>7}{ia:>6}{ib:>5}")


def _ago(ts: float) -> str:
    s = time.time() - ts
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{int(s // n)}{unit} ago"
    return "just now"


def _model_lines(models: list[ModelInfo], history: dict, base_url: str, numbered: bool = False) -> list[str]:
    width = max([len(m.id) for m in models] + [10])
    lines = []
    for i, m in enumerate(models, 1):
        h = history.get((base_url, m.id))
        last = ""
        if h and h["latest"]:
            avg = h["latest"]["avg_score"]
            last = f"last run {_ago(h['latest']['started_at'])}" + (f", avg {avg:.2f}" if avg is not None else "")
        elif h:
            last = f"{h['runs']} run(s)"
        else:
            last = "never tested"
        idx = f"{i:>3}) " if numbered else "  "
        col = C["dim"] if not m.chat_capable else ""
        lines.append(f"{idx}{col}{m.id:<{width}}{C['reset']}  {C['dim']}{m.describe() or '-'}{C['reset']}  "
                     f"{C['yellow'] if last == 'never tested' else C['dim']}{last}{C['reset']}")
    return lines


async def _discover(cfg, store) -> list[ModelInfo]:
    try:
        models = await discover(cfg.target)
    except DiscoveryError as e:
        sys.exit(str(e))
    store.save_models(cfg.target.base_url, [m.to_dict() for m in models])
    return models


def _parse_selection(text: str, n: int) -> list[int] | None:
    text = text.strip().lower()
    if text == "all":
        return list(range(n))
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        if not lo.isdigit() or (hi and not hi.isdigit()):
            return None
        for k in range(int(lo), int(hi or lo) + 1):
            if not 1 <= k <= n:
                return None
            if k - 1 not in out:
                out.append(k - 1)
    return out


async def _choose_models(cfg, store, interactive: bool) -> list[str]:
    """Pull the endpoint's model list and resolve which models to test."""
    spec = cfg.target.model
    models = await _discover(cfg, store)
    chat = [m for m in models if m.chat_capable]
    if spec:
        chosen = match_models(spec, models)
        known = {m.id for m in models}
        for m in chosen:
            if m not in known:
                print(f"{C['yellow']}note:{C['reset']} '{m}' is not in the server's model list; trying anyway")
        if not chosen:
            sys.exit(f"no models at {cfg.target.base_url} match '{spec}'. Available: "
                     + ", ".join(m.id for m in chat))
        return chosen
    if not chat:
        sys.exit(f"no chat models available at {cfg.target.base_url}")
    if len(chat) == 1 or not interactive:
        if len(chat) > 1:
            print(f"{C['dim']}{len(chat)} models available; using {chat[0].id} "
                  f"(choose with -m <id|glob|all>){C['reset']}")
        return [chat[0].id]
    history = store.model_history()
    print(f"{C['bold']}Models at {cfg.target.base_url}{C['reset']}")
    print("\n".join(_model_lines(chat, history, cfg.target.base_url, numbered=True)))
    while True:
        try:
            raw = input(f"Test which? number, list (1,3), range (1-3) or 'all' [1]: ") or "1"
        except EOFError:
            raw = "1"
        sel = _parse_selection(raw, len(chat))
        if sel:
            return [chat[i].id for i in sel]
        print("didn't understand that selection")


async def cmd_models(a):
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    models = await _discover(cfg, store)
    if a.json:
        print(json.dumps([m.to_dict() for m in models], indent=2))
        return
    if not models:
        print(f"no models at {cfg.target.base_url}")
        return
    server = next((m.server for m in models if m.server != "openai"), "openai-compatible")
    print(f"{C['bold']}{len(models)} model(s) at {cfg.target.base_url}{C['reset']} {C['dim']}({server}){C['reset']}")
    print("\n".join(_model_lines(models, store.model_history(), cfg.target.base_url)))


def print_prompt(p: PromptInfo, template: bool = False):
    head = f"{C['bold']}{p.model}{C['reset']}" + (f" {C['dim']}({p.server}){C['reset']}" if p.server else "")
    print(head)
    if not p.available:
        print(f"{C['dim']}{p.note}{C['reset']}")
        return
    if p.system:
        label = "built-in system prompt" + (" (everything the template adds before your first message)"
                                            if p.server == "llamacpp" else "")
        print(f"\n{C['bold']}{label}{C['reset']}\n{p.system.rstrip()}")
    elif p.system == "":
        print(f"\n{C['bold']}built-in system prompt:{C['reset']} none. Without a system message from tealeaf, the "
              "model sees only the conversation.")
    if p.rendered:
        print(f"\n{C['bold']}rendered prompt for one user message{C['reset']}\n{C['dim']}{p.rendered}{C['reset']}")
    if p.note:
        print(f"\n{C['yellow']}note:{C['reset']} {p.note}")
    if template and p.template:
        print(f"\n{C['bold']}chat template{C['reset']}\n{p.template}")
    elif p.template:
        print(f"\n{C['dim']}chat template: {len(p.template):,} chars (show it with --template){C['reset']}")


async def cmd_prompt(a):
    if a.run_id:
        store = Store(load_config(a.config).db_path)
        run = _find_run(store, a.run_id)
        if not run:
            sys.exit("run not found")
        snap = (run.get("config") or {}).get("model_prompt")
        if not snap:
            sys.exit(f"run {run['id']} has no prompt snapshot (runs before this feature, or the server was unreachable)")
        p = PromptInfo(**snap)
    else:
        cfg = _apply_overrides(load_config(a.config), a)
        store = Store(cfg.db_path)
        model = (await _choose_models(cfg, store, interactive=sys.stdin.isatty()))[0]
        p = await model_prompt(cfg.target, model)
    if a.json:
        print(json.dumps(p.to_dict(), indent=2))
    else:
        print_prompt(p, a.template)


def _pm(sp: dict | None, digits: int = 2) -> str:
    if not sp:
        return "-"
    return f"{sp['mean']:.{digits}f}±{sp['sd']:.{digits}f}"


def print_stability(rep: dict, brief: bool = False):
    runs = rep["runs"]
    print(f"\n{C['bold']}Stability across {len(runs)} runs{C['reset']} {C['dim']}"
          f"({', '.join(rep['models'])}; {rep['key'] if len(rep['key']) < 40 else 'selected runs'}){C['reset']}")
    for w in rep["warnings"]:
        print(f"{C['yellow']}warning:{C['reset']} {w}")
    if rep.get("judge_model"):
        print(f"{C['dim']}belief consistency judged by {rep['judge_model']}{C['reset']}")
    cases = rep["cases"]
    if not cases:
        print("no finished cases to compare")
        return
    width = min(44, max(len(c["case_id"]) for c in cases))
    print(f"{'case':<{width}}  {'stab':>5}  {'n':>2}  {'statuses':<22} {'score':>11}  {'interest':>10}")
    for c in (cases[:12] if brief else cases):
        st = ", ".join(f"{k} {v}" for k, v in sorted(c["statuses"].items(), key=lambda kv: -kv[1]))
        sv = c["stability"]
        col = "" if sv is None else C["fail"] if sv < 0.5 else C["yellow"] if sv < 0.8 else ""
        stab = "   ?" if sv is None else f"{sv:.2f}"
        print(f"{c['case_id'][:width]:<{width}}  {col}{stab:>5}{C['reset']}  {c['n']:>2}  {st:<22} "
              f"{_pm(c['score']):>11}  {_pm(c['interest'], 1):>10}")
        if brief:
            continue
        j = c.get("judge") or {}
        if j.get("summary"):
            print(f"    {C['dim']}{j['summary']}{C['reset']}")
        for u in (j.get("unstable") or [])[:4]:
            if isinstance(u, dict):
                variants = " | ".join(str(v) for v in (u.get("variants") or [])[:4])
                print(f"    {C['yellow']}varies:{C['reset']} {u.get('topic')}: {C['dim']}{variants}{C['reset']}")
        if c["flags_always"]:
            print(f"    {C['dim']}flagged every run: {', '.join(c['flags_always'][:6])}{C['reset']}")
    if any(c["stability"] is None for c in cases) and not rep.get("judge_model"):
        print(f"{C['dim']}? = probe: its stability needs the judge to compare beliefs across runs (--judge){C['reset']}")
    if brief and len(cases) > 12:
        print(f"{C['dim']}… {len(cases) - 12} more (least stable shown first){C['reset']}")


async def cmd_stability(a):
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    refs = a.refs
    if not refs:
        groups = store.groups(1)
        if not groups:
            sys.exit("no repeat groups yet: run with `tealeaf run --repeat N`, or pass two or more run ids")
        refs = [groups[0]["id"]]
    try:
        key, ids = resolve_runs(store, refs)
    except ValueError as e:
        sys.exit(str(e))
    if a.judge:
        jm = cfg.judge_endpoint()
        print(f"{C['dim']}asking {jm.model or 'the judge'} to compare beliefs across {len(ids)} runs…{C['reset']}")
        await judge_consistency(store, key, ids, jm,
                                on_case=lambda cid, d: print(f"{CLR}  {cid}: "
                                                             f"{(d or {}).get('consistency', '?')}/10", flush=True))
    rep = compute(store, key, ids)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print_stability(rep)


async def cmd_resume(a):
    """Continue a cancelled/interrupted run in place, with the settings it started with."""
    now = load_config(a.config)
    store = Store(now.db_path)
    store.mark_stale_runs()
    run = _find_run(store, a.run_id)
    if not run:
        sys.exit("run not found")
    if run.get("config", {}).get("reannotated_from"):
        sys.exit("this is a re-annotation run; run `tealeaf annotate` on the source run again instead")
    cfg = restore_config(now, run.get("config") or {})
    if a.api_key_env:  # after restoring: an explicit key is meant for the run's own server, wherever that is
        cfg.target.api_key = _key_from_env(a.api_key_env)
    if a.opik:
        cfg.opik.enabled = True
    results = store.get_results(run["id"])
    todo = [r for r in results if r["status"] in ("pending", "running", "skipped") or
            (a.errors and r["status"] == "error")]
    errors = sum(1 for r in results if r["status"] == "error")
    if not todo:
        hint = f" ({errors} errored case(s): add --errors to rerun them)" if errors and not a.errors else ""
        sys.exit(f"run {run['id']} has nothing left to run{hint}")
    jm = "(self)" if cfg.judge is None else cfg.judge.model
    print(f"{C['bold']}resuming run {run['id']}{C['reset']}  target={cfg.target.model} @ {cfg.target.base_url}  "
          f"judge={jm}  cases={len(todo)} of {len(results)}")
    opik_make = _opik_factory(cfg)
    runner = Runner(cfg, store, opik=opik_make() if opik_make else None, listeners=[Printer(len(todo), a.verbose)])
    try:
        await runner.resume(run["id"], include_errors=a.errors)
    except ClientError as e:
        sys.exit(f"cannot reach target: {e}")
    except ValueError as e:
        sys.exit(str(e))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted; run `tealeaf resume` again to continue")
    print_summary(store.get_run(run["id"]))
    left = [r for r in store.get_results(run["id"]) if r["status"] == "error"]
    if left and not a.errors:
        print(f"{C['dim']}{len(left)} case(s) errored; `tealeaf resume {run['id']} --errors` reruns them{C['reset']}")


async def cmd_chat(a):
    """Interactive chat with the target; saved as a run so it appears in the dashboard."""
    from .registry import Case
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    picked = await _choose_models(cfg, store, interactive=sys.stdin.isatty())
    if len(picked) > 1:
        print(f"{C['dim']}chat uses one model; using {picked[0]}{C['reset']}")
    cfg.target.model = picked[0]
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


async def cmd_annotate(a):
    """Judge an existing run's transcripts again (e.g. with a different judge) as a new run."""
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    src = _find_run(store, a.run_id)
    if not src:
        sys.exit("run not found")
    if src["status"] in ("running", "queued", "cancelling"):
        sys.exit(f"run {src['id']} is still {src['status']}; wait for it to finish")
    ids = [r["case_id"] for r in store.get_results(src["id"])]
    if a.case:
        ids = [i for i in ids if any(fnmatch.fnmatchcase(i, pat) for pat in a.case)]
        if not ids:
            sys.exit("no cases in that run match the selection")
    # The judge-is-target fallback must judge with the model that produced the run, not today's [target].
    # Today's target key goes along only if the source run used the same server.
    cfg.target = replace(cfg.target, base_url=src["target_url"], model=src["target_model"], extra={},
                         api_key=key_for(src["target_url"], cfg.target))
    runner = Runner(cfg, store, listeners=[Printer(len(ids), False)])
    jm = "(self)" if runner.judge_is_target else runner.judge_client.model
    print(f"{C['bold']}re-annotating run {src['id']}{C['reset']}  ({src['target_model']}, was judged by "
          f"{src['judge_model']})  judge={jm}  cases={len(ids)}")
    try:
        new_id = await runner.reannotate(src["id"], ids, a.label or "")
    except ClientError as e:
        sys.exit(f"cannot reach judge: {e}")
    print_summary(store.get_run(new_id))
    print(f"\ncompare: tealeaf compare {src['id']} {new_id}")


def cmd_steer_agent(a):
    """Run on the llama.cpp machine: builds control vectors and serves a steered llama-server for `tealeaf steer`."""
    import secrets
    import shlex

    import uvicorn

    from .steer_agent import Agent, AgentFailure, create_agent_app
    token = a.token or os.environ.get("TEALEAF_STEER_TOKEN", "")
    if not token:
        token = secrets.token_urlsafe(24)
        print(f"no --token or TEALEAF_STEER_TOKEN given; generated one for this session:\n  {token}\n"
              f"put it in TEALEAF_STEER_TOKEN where you run `tealeaf steer`", flush=True)
    try:
        agent = Agent(bin_dir=a.llama_bin, models_dir=a.models_dir, default_model=a.model, data_dir=a.data,
                      server_host=a.server_host, server_port=a.server_port,
                      server_args=shlex.split(a.server_args or ""), gen_args=shlex.split(a.gen_args or ""))
    except AgentFailure as e:
        sys.exit(f"steer-agent: {e}")
    print(f"steer-agent on {a.host}:{a.port}  (steered llama-server will use port {a.server_port})\n"
          f"models: {agent.models_dir}  vectors: {agent.data_dir / 'vectors'}", flush=True)
    uvicorn.run(create_agent_app(agent, token), host=a.host, port=a.port, log_level="warning")


def print_steering(rep: dict, show_all: bool = False):
    from .steering import scale_key
    steps = rep["steps"]
    print(f"\n{C['bold']}Steering: {rep['concept']}{C['reset']} {C['dim']}({rep['negative']} ← → {rep['positive']}; "
          f"{rep['target_model']}; sweep {rep['id']}){C['reset']}")
    heads = [f"{s['scale']:+g}" for s in steps]
    width = min(44, max([len(c["case_id"]) for c in rep["cases"]] + [10]))
    print(f"{'':<{width}}  " + "".join(f"{h:>8}" for h in heads) + f"{'range':>8}")

    def row(name, vals, extra=""):
        cells = "".join(f"{'-' if v is None else f'{v:.2f}':>8}" for v in vals)
        print(f"{name[:width]:<{width}}  {cells}{extra}")

    for suite, d in rep["suites"].items():
        row(f"[{suite}]", [d.get(scale_key(s["scale"])) for s in steps])
    print()
    cases = rep["cases"] if show_all else rep["cases"][:20]
    for c in cases:
        vals = [(c["by_scale"].get(scale_key(s["scale"])) or {}).get("score") for s in steps]
        rng = c["range"]
        col = C["yellow"] if rng is not None and rng >= 0.3 else ""
        row(c["case_id"], vals, f"{col}{'-' if rng is None else f'{rng:.2f}':>8}{C['reset']}")
    if len(rep["cases"]) > len(cases):
        print(f"{C['dim']}… {len(rep['cases']) - len(cases)} more (--all){C['reset']}")
    print(f"{C['dim']}cases most moved by the vector first. Details: tealeaf show <run> <case>; runs: "
          + ", ".join(f"{s['scale']:+g}={s['run_id']}" for s in steps) + C["reset"])


async def cmd_steer(a):
    from .steering import CONCEPTS, AgentClient, AgentError, parse_scales, report, sweep
    if a.list or not a.concept:
        for c in CONCEPTS.values():
            print(f"{C['bold']}{c.name:<12}{C['reset']} {c.negative} ← → {c.positive}  {C['dim']}{c.description}\n"
                  f"{'':<13}cases: {', '.join(c.cases)}{C['reset']}")
        return
    concept = CONCEPTS.get(a.concept)
    if not concept:
        sys.exit(f"unknown concept {a.concept!r}; known: {', '.join(CONCEPTS)}")
    cfg = _apply_overrides(load_config(a.config), a)
    store = Store(cfg.db_path)
    store.mark_stale_runs()
    try:
        scales = parse_scales(a.scales or cfg.steer.scales)
        cases = select_cases(patterns=a.case) if a.case else None
        layers = [int(x) for x in a.layers.split("-", 1)] if a.layers else None
    except ValueError as e:
        sys.exit(f"bad option: {e}")
    try:
        agent = AgentClient(a.agent or cfg.steer.agent_url, cfg.steer.token)
    except AgentError as e:
        sys.exit(str(e))
    total = len(cases or select_cases(patterns=list(concept.cases)))
    print(f"{C['bold']}steering {concept.name}{C['reset']} ({concept.negative} ← → {concept.positive}) at "
          f"{', '.join(f'{x:+g}' for x in scales)} · {total} cases per strength · agent {agent.url}")
    try:
        sid = await sweep(cfg, store, agent, concept, scales, cases=cases, gguf=a.gguf, layers=layers,
                          method=a.method, rebuild=a.rebuild, label=a.label or "",
                          listeners=[lambda: Printer(total, a.verbose)], keep_server=a.keep_server,
                          on_status=lambda m: print(f"{C['dim']}{m}{C['reset']}", flush=True))
    except (AgentError, ValueError) as e:
        sys.exit(str(e))
    finally:
        await agent.aclose()
    print_steering(report(store, sid))


def cmd_steering(a):
    from .steering import report
    store = Store(load_config(a.config).db_path)
    sweeps = store.sweeps()
    if not sweeps:
        sys.exit("no steering sweeps yet: `tealeaf steer <concept>`")
    sid = a.sweep or sweeps[0]["id"]
    match = next((s["id"] for s in sweeps if s["id"] == sid or s["id"].startswith(sid)), None)
    if not match:
        sys.exit(f"no steering sweep matches {sid!r}")
    rep = report(store, match)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print_steering(rep, a.all)


def cmd_mock(a):
    import uvicorn

    from .mock_server import create_mock_app
    print(f"mock {a.flavor} server: http://{a.host}:{a.port}/v1")
    uvicorn.run(create_mock_app(a.delay, a.flavor), host=a.host, port=a.port, log_level="warning")


# --------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tealeaf", description="Tealeaf, a local LLM test harness: benchmarks + behavioral probes.")
    p.add_argument("--config", help="path to tealeaf.toml (default: ./tealeaf.toml, ~/.config/tealeaf/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def target_args(sp):
        sp.add_argument("--base-url", help="target OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
        sp.add_argument("-m", "--model", help="model id, comma list, glob ('qwen*') or 'all' "
                                              "(default: pick interactively, or the first listed)")
        sp.add_argument("--api-key-env", metavar="VAR", help="read the target's API key from this environment "
                        "variable, e.g. OPENROUTER_API_KEY (default: TEALEAF_API_KEY, else [target] api_key)")
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
    r.add_argument("--contamination-file", action="append", metavar="JSONL",
                   help="benchmark file for contamination.your_items (repeatable)")
    r.add_argument("--opik", action="store_true", help="also log traces to Opik")
    r.add_argument("-v", "--verbose", action="store_true", help="print transcripts as they happen")
    r.add_argument("--web", action="store_true", help="serve the dashboard while running")
    r.add_argument("--repeat", type=int, help="run the selection N times per model, grouped for `tealeaf stability`")
    r.add_argument("--system", dest="run_system", metavar="TEXT",
                   help="system prompt put at the start of every conversation with the target")
    r.add_argument("--system-file", dest="run_system_file", metavar="PATH", help="read --system from a file")
    r.add_argument("-y", "--yes", action="store_true", help="don't prompt for models; use the first one")
    r.add_argument("--host", help="address to listen on (default: [web] host, else 127.0.0.1; 0.0.0.0 = all interfaces)")
    r.add_argument("--port", type=int, help="default: [web] port, else 8765")

    s = sub.add_parser("serve", help="serve the web dashboard (and start runs from it)")
    target_args(s)
    s.add_argument("--opik", action="store_true")
    s.add_argument("--host", help="address to listen on (default: [web] host, else 127.0.0.1; 0.0.0.0 = all interfaces)")
    s.add_argument("--port", type=int, help="default: [web] port, else 8765")

    an = sub.add_parser("annotate", help="judge an existing run's transcripts again, as a new run "
                                         "(e.g. to compare judges); the target model is not called")
    an.add_argument("run_id", help="run id (prefix ok)")
    an.add_argument("-c", "--case", action="append", help="case id glob, e.g. 'beliefs.*' (repeatable)")
    an.add_argument("--judge-url", help="judge base URL (default: [judge] in tealeaf.toml)")
    an.add_argument("--judge-model", help="judge model id (default: [judge] in tealeaf.toml)")
    an.add_argument("-l", "--label", help="label for the new run")

    re_ = sub.add_parser("resume", help="continue a cancelled or interrupted run in place, with its original settings")
    re_.add_argument("run_id", help="run id (prefix ok)")
    re_.add_argument("--errors", action="store_true", help="also rerun cases that errored (e.g. timeouts)")
    re_.add_argument("--api-key-env", metavar="VAR", help="read the target's API key from this environment variable "
                     "(keys aren't stored with the run)")
    re_.add_argument("--opik", action="store_true")
    re_.add_argument("-v", "--verbose", action="store_true", help="print transcripts as they happen")

    li = sub.add_parser("list", help="list suites and cases")
    li.add_argument("-k", "--kind", action="append", choices=["traditional", "probe"])
    li.add_argument("--brief", action="store_true")

    rs = sub.add_parser("runs", help="list past runs")
    rs.add_argument("-n", "--limit", type=int, default=20)

    sh = sub.add_parser("show", help="show a run, or one case's transcript")
    sh.add_argument("run_id", help="run id (prefix ok)")
    sh.add_argument("case", nargs="?", help="case id, e.g. beliefs.identity")
    sh.add_argument("--tokens", action="store_true",
                    help="mark uncertain tokens with their probability and the runner-up (needs logprobs)")

    cp = sub.add_parser("compare", help="compare scores of two runs")
    cp.add_argument("run_a")
    cp.add_argument("run_b")

    mo = sub.add_parser("models", help="list models at the target endpoint, with metadata and test history")
    target_args(mo)
    mo.add_argument("--json", action="store_true")

    st = sub.add_parser("stability", help="how much each case varies across repeated runs")
    st.add_argument("refs", nargs="*", help="a repeat group id, or two or more run ids (default: latest group)")
    st.add_argument("--judge", action="store_true",
                    help="also have the judge compare each probe's beliefs across runs (one judge call per case)")
    st.add_argument("--judge-url", help="judge base URL (default: [judge] in tealeaf.toml)")
    st.add_argument("--judge-model", help="judge model id (default: [judge] in tealeaf.toml)")
    st.add_argument("--json", action="store_true")

    pr = sub.add_parser("prompt", help="show the model's built-in system prompt and chat template "
                                       "(llama.cpp and Ollama expose them; hosted APIs don't)")
    target_args(pr)
    pr.add_argument("--run", dest="run_id", help="show the snapshot saved with this run instead (prefix ok)")
    pr.add_argument("--template", action="store_true", help="also print the full chat template")
    pr.add_argument("--json", action="store_true")

    ch = sub.add_parser("chat", help="interactive conversation with the target (saved + annotated)")
    target_args(ch)
    ch.add_argument("--system", help="system prompt")
    ch.add_argument("-l", "--label")
    ch.add_argument("--show-reasoning", action="store_true")

    sa = sub.add_parser("steer-agent", help="run on the llama.cpp machine: builds control vectors and serves a "
                                             "steered llama-server for `tealeaf steer`")
    sa.add_argument("--models-dir", required=True, help="folder with the GGUF files it may load")
    sa.add_argument("--model", help="default GGUF (a path inside --models-dir)")
    sa.add_argument("--llama-bin", help="folder with llama-server and llama-cvector-generator (default: PATH)")
    sa.add_argument("--token", help="shared secret (default: TEALEAF_STEER_TOKEN, else one is generated)")
    sa.add_argument("--host", default="0.0.0.0", help="agent address (default 0.0.0.0; the token guards it)")
    sa.add_argument("--port", type=int, default=8766)
    sa.add_argument("--server-host", default="0.0.0.0", help="address for the steered llama-server")
    sa.add_argument("--server-port", type=int, default=8081, help="port for the steered llama-server")
    sa.add_argument("--server-args", help="extra llama-server flags, e.g. \"-ngl 99 -c 8192\"")
    sa.add_argument("--gen-args", help="extra llama-cvector-generator flags, e.g. \"-ngl 99\"")
    sa.add_argument("--data", default="~/.local/share/tealeaf/steer", help="where vectors and logs go")

    se = sub.add_parser("steer", help="build a steering vector for a concept and run its probes at several "
                                      "strengths (needs `tealeaf steer-agent` on the llama.cpp machine)")
    target_args(se)
    se.add_argument("concept", nargs="?", help="emotion, honesty, sycophancy or ai_identity (--list)")
    se.add_argument("--list", action="store_true", help="list the concepts and the cases each one measures")
    se.add_argument("--scales", help="comma list of strengths, e.g. -1.5,-0.75,0,0.75,1.5 (default: [steer] scales)")
    se.add_argument("-c", "--case", action="append", help="case globs instead of the concept's own (repeatable)")
    se.add_argument("--gguf", help="GGUF on the agent (default: the agent's --model)")
    se.add_argument("--layers", metavar="A-B", help="apply the vector to these layers only, e.g. 10-20")
    se.add_argument("--method", choices=["pca", "mean"], default="pca", help="how llama-cvector-generator builds it")
    se.add_argument("--rebuild", action="store_true", help="build the vector again even if the agent has it")
    se.add_argument("--agent", help="agent URL (default: [steer] agent_url)")
    se.add_argument("--keep-server", action="store_true", help="leave the steered llama-server running afterwards")
    se.add_argument("-l", "--label")
    se.add_argument("--no-annotate", action="store_true", help="skip judge annotation of probes")
    se.add_argument("-v", "--verbose", action="store_true", help="print transcripts as they happen")

    sg = sub.add_parser("steering", help="show a steering sweep: each case's score at each strength")
    sg.add_argument("sweep", nargs="?", help="sweep id or prefix (default: the latest)")
    sg.add_argument("--all", action="store_true", help="every case, not just the 20 most moved")
    sg.add_argument("--json", action="store_true")

    mk = sub.add_parser("mock-server", help="run a fake OpenAI-compatible model for development")
    mk.add_argument("--host", default="127.0.0.1")
    mk.add_argument("--port", type=int, default=11999)
    mk.add_argument("--delay", type=float, default=0.01, help="seconds between streamed tokens")
    mk.add_argument("--flavor", choices=["ollama", "lmstudio", "llamacpp", "openai"], default="ollama",
                    help="which server's native model API to imitate")
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
        elif a.cmd == "resume":
            asyncio.run(cmd_resume(a))
        elif a.cmd == "stability":
            asyncio.run(cmd_stability(a))
        elif a.cmd == "prompt":
            asyncio.run(cmd_prompt(a))
        elif a.cmd == "chat":
            asyncio.run(cmd_chat(a))
        elif a.cmd == "annotate":
            asyncio.run(cmd_annotate(a))
        elif a.cmd == "mock-server":
            cmd_mock(a)
        elif a.cmd == "steer-agent":
            cmd_steer_agent(a)
        elif a.cmd == "steer":
            asyncio.run(cmd_steer(a))
        elif a.cmd == "steering":
            cmd_steering(a)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
