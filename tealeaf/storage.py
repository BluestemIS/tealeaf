"""SQLite persistence. The DB is the bus between the runner (CLI or web) and the dashboard."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    label TEXT,
    status TEXT,            -- running | cancelling | cancelled | finished | error | interrupted
    started_at REAL,
    finished_at REAL,
    pid INTEGER,
    target_model TEXT,
    target_url TEXT,
    judge_model TEXT,
    judge_is_target INTEGER,
    total_cases INTEGER,
    done_cases INTEGER DEFAULT 0,
    config TEXT,
    summary TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    case_id TEXT,
    suite TEXT,
    kind TEXT,
    title TEXT,
    status TEXT,            -- pending | running | pass | fail | observed | error | skipped
    score REAL,
    interest REAL,
    flags TEXT,
    metrics TEXT,
    notes TEXT,
    judge TEXT,
    error TEXT,
    started_at REAL,
    finished_at REAL,
    UNIQUE(run_id, case_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    case_id TEXT,
    thread TEXT,
    role TEXT,
    content TEXT,
    reasoning TEXT,
    meta TEXT,
    ts REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    ts REAL,
    type TEXT,
    key TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS models (
    base_url TEXT,
    model_id TEXT,
    info TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (base_url, model_id)
);
CREATE TABLE IF NOT EXISTS reports (   -- cached derived analyses, e.g. kind='stability', key=group id
    kind TEXT,
    key TEXT,
    created_at REAL,
    data TEXT,
    PRIMARY KEY (kind, key)
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_messages_case ON messages(run_id, case_id);
CREATE INDEX IF NOT EXISTS idx_events_key ON events(key);
"""

_JSON_COLS = {"data", "info", "config", "summary", "flags", "metrics", "notes", "judge", "meta", "payload"}


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in _JSON_COLS & d.keys():
        if d[k]:
            try:
                d[k] = json.loads(d[k])
            except (TypeError, json.JSONDecodeError):
                pass
    return d


def _j(v):
    return json.dumps(v, default=str) if v is not None else None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [_row(r) for r in self._db.execute(sql, args).fetchall()]

    # ---- runs -------------------------------------------------------------
    def create_run(self, *, label, target_model, target_url, judge_model, judge_is_target, total_cases, config,
                   status: str = "running") -> str:
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self._exec(
            "INSERT INTO runs (id,label,status,started_at,pid,target_model,target_url,judge_model,judge_is_target,"
            "total_cases,config) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, label, status, time.time(), os.getpid(), target_model, target_url, judge_model,
             int(judge_is_target), total_cases, _j(config)),
        )
        return run_id

    def update_run(self, run_id: str, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [_j(v) if k in _JSON_COLS else v for k, v in fields.items()]
        self._exec(f"UPDATE runs SET {cols} WHERE id=?", (*vals, run_id))

    def get_run(self, run_id: str) -> dict | None:
        rows = self._q("SELECT * FROM runs WHERE id=?", (run_id,))
        return rows[0] if rows else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        return self._q(
            "SELECT id,label,status,started_at,finished_at,target_model,target_url,judge_model,judge_is_target,"
            "total_cases,done_cases,summary,error FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))

    def delete_run(self, run_id: str):
        for t in ("messages", "results", "events"):
            self._exec(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
        self._exec("DELETE FROM runs WHERE id=?", (run_id,))

    def request_cancel(self, run_id: str):
        self._exec("UPDATE runs SET status='cancelling' WHERE id=? AND status='running'", (run_id,))
        self._exec("UPDATE runs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                   (time.time(), run_id))
        self._exec("UPDATE results SET status='skipped' WHERE run_id=? AND status='pending' AND "
                   "(SELECT status FROM runs WHERE id=?)='cancelled'", (run_id, run_id))

    def cancel_requested(self, run_id: str) -> bool:
        rows = self._q("SELECT status FROM runs WHERE id=?", (run_id,))
        return bool(rows) and rows[0]["status"] == "cancelling"

    def mark_stale_runs(self):
        """Runs whose owning process died are marked interrupted."""
        for r in self._q("SELECT id,pid FROM runs WHERE status IN ('queued','running','cancelling')"):
            if not _pid_alive(r["pid"]):
                self.update_run(r["id"], status="interrupted", finished_at=time.time())
                self._exec("UPDATE results SET status='skipped' WHERE run_id=? AND status IN ('pending','running')",
                           (r["id"],))

    # ---- models -----------------------------------------------------------
    def save_models(self, base_url: str, models: list[dict]):
        now = time.time()
        for m in models:
            self._exec(
                "INSERT INTO models (base_url,model_id,info,first_seen,last_seen) VALUES (?,?,?,?,?) "
                "ON CONFLICT(base_url,model_id) DO UPDATE SET info=excluded.info, last_seen=excluded.last_seen",
                (base_url, m["id"], _j(m), now, now))

    def known_models(self, base_url: str | None = None) -> list[dict]:
        if base_url:
            return self._q("SELECT * FROM models WHERE base_url=? ORDER BY model_id", (base_url,))
        return self._q("SELECT * FROM models ORDER BY base_url, model_id")

    def model_history(self) -> dict[tuple[str, str], dict]:
        """Latest finished run + run count per (base_url, model)."""
        out: dict[tuple[str, str], dict] = {}
        for r in self._q("SELECT id,label,status,started_at,target_url,target_model,summary FROM runs "
                         "WHERE status NOT IN ('queued') ORDER BY started_at DESC"):
            k = (r["target_url"], r["target_model"])
            h = out.setdefault(k, {"runs": 0, "latest": None})
            h["runs"] += 1
            if h["latest"] is None and r["status"] in ("finished", "cancelled"):
                s = r.get("summary") or {}
                h["latest"] = {"id": r["id"], "label": r["label"], "started_at": r["started_at"],
                               "avg_score": s.get("avg_score") if isinstance(s, dict) else None,
                               "tokens_per_sec": s.get("tokens_per_sec") if isinstance(s, dict) else None}
        return out

    # ---- results ----------------------------------------------------------
    def create_result(self, run_id, case) -> int:
        cur = self._exec(
            "INSERT INTO results (run_id,case_id,suite,kind,title,status) VALUES (?,?,?,?,?,?)",
            (run_id, case.id, case.suite, case.kind, case.title, "pending"),
        )
        return cur.lastrowid

    def update_result(self, run_id: str, case_id: str, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [_j(v) if k in _JSON_COLS else v for k, v in fields.items()]
        self._exec(f"UPDATE results SET {cols} WHERE run_id=? AND case_id=?", (*vals, run_id, case_id))

    def clear_result(self, run_id: str, case_id: str):
        """Reset a case to pending and drop its messages, so it can run again (resume)."""
        self._exec("DELETE FROM messages WHERE run_id=? AND case_id=?", (run_id, case_id))
        self.update_result(run_id, case_id, status="pending", score=None, interest=None, flags=None, metrics=None,
                           notes=None, judge=None, error=None, started_at=None, finished_at=None)

    def get_results(self, run_id: str) -> list[dict]:
        return self._q("SELECT * FROM results WHERE run_id=? ORDER BY id", (run_id,))

    def get_result(self, run_id: str, case_id: str) -> dict | None:
        rows = self._q("SELECT * FROM results WHERE run_id=? AND case_id=?", (run_id, case_id))
        return rows[0] if rows else None

    def highlights(self, min_interest: float = 0, limit: int = 200) -> list[dict]:
        return self._q(
            "SELECT r.run_id,r.case_id,r.suite,r.title,r.status,r.score,r.interest,r.flags,r.judge,"
            "runs.target_model,runs.started_at FROM results r JOIN runs ON runs.id=r.run_id "
            "WHERE r.interest IS NOT NULL AND r.interest>=? ORDER BY r.interest DESC, runs.started_at DESC LIMIT ?",
            (min_interest, limit))

    def annotated(self) -> list[dict]:
        return self._q(
            "SELECT r.run_id,r.case_id,r.title,r.judge,runs.target_model,runs.started_at FROM results r "
            "JOIN runs ON runs.id=r.run_id WHERE r.judge IS NOT NULL ORDER BY runs.started_at DESC")

    # ---- messages ---------------------------------------------------------
    def add_message(self, run_id, case_id, thread, role, content, reasoning="", meta=None) -> dict:
        ts = time.time()
        cur = self._exec(
            "INSERT INTO messages (run_id,case_id,thread,role,content,reasoning,meta,ts) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, case_id, thread, role, content, reasoning, _j(meta or {}), ts),
        )
        return {"id": cur.lastrowid, "run_id": run_id, "case_id": case_id, "thread": thread, "role": role,
                "content": content, "reasoning": reasoning, "meta": meta or {}, "ts": ts}

    def get_messages(self, run_id: str, case_id: str) -> list[dict]:
        return self._q("SELECT * FROM messages WHERE run_id=? AND case_id=? ORDER BY id", (run_id, case_id))

    # ---- events -----------------------------------------------------------
    def add_event(self, run_id: str, type_: str, payload: dict, key: str | None = None) -> int:
        cur = self._exec("INSERT INTO events (run_id,ts,type,key,payload) VALUES (?,?,?,?,?)",
                         (run_id, time.time(), type_, key, _j(payload)))
        return cur.lastrowid

    def delete_events_by_key(self, key: str):
        self._exec("DELETE FROM events WHERE key=?", (key,))

    def events_after(self, after_id: int, limit: int = 500) -> list[dict]:
        return self._q("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (after_id, limit))

    def last_event_id(self) -> int:
        rows = self._q("SELECT COALESCE(MAX(id),0) AS m FROM events")
        return rows[0]["m"]

    def prune_events(self, keep_seconds: float = 86400):
        self._exec("DELETE FROM events WHERE ts<?", (time.time() - keep_seconds,))

    # ---- repeat groups / reports -------------------------------------------
    def groups(self, limit: int = 100) -> list[dict]:
        """Series of repeated runs (`tealeaf run --repeat N`), newest first."""
        rows = self._q(
            "SELECT json_extract(config,'$.group.id') AS gid, id, label, status, started_at, target_model, "
            "target_url, json_extract(config,'$.run.system_prompt') AS system_prompt FROM runs "
            "WHERE json_extract(config,'$.group.id') IS NOT NULL ORDER BY started_at")
        out: dict[str, dict] = {}
        for r in rows:
            g = out.setdefault(r["gid"], {"id": r["gid"], "label": r["label"], "target_model": r["target_model"],
                                          "target_url": r["target_url"], "system_prompt": r["system_prompt"],
                                          "started_at": r["started_at"], "runs": [], "statuses": []})
            g["runs"].append(r["id"])
            g["statuses"].append(r["status"])
        return sorted(out.values(), key=lambda g: -g["started_at"])[:limit]

    def group_run_ids(self, group_id: str) -> list[str]:
        return [r["id"] for r in self._q("SELECT id FROM runs WHERE json_extract(config,'$.group.id')=? "
                                         "ORDER BY started_at", (group_id,))]

    def save_report(self, kind: str, key: str, data: dict):
        self._exec("INSERT INTO reports (kind,key,created_at,data) VALUES (?,?,?,?) ON CONFLICT(kind,key) "
                   "DO UPDATE SET created_at=excluded.created_at, data=excluded.data", (kind, key, time.time(), _j(data)))

    def get_report(self, kind: str, key: str) -> dict | None:
        rows = self._q("SELECT * FROM reports WHERE kind=? AND key=?", (kind, key))
        return rows[0] if rows else None
