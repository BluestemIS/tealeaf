"""Re-annotating a finished run with another judge, and the dashboard's listen address."""
import argparse
import os
import asyncio

from tealeaf.cli import _listen
from tealeaf.config import load_config
from tealeaf.registry import select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store


def _cfg(monkeypatch, tmp_path, url):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_BASE_URL", url)
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    cfg = load_config()
    cfg.run.conversation_turns = 2
    return cfg


def test_reannotate_copies_transcripts_and_rejudges(mock_url, tmp_path, monkeypatch):
    cfg = _cfg(monkeypatch, tmp_path, mock_url)
    store = Store(cfg.db_path)
    cases = select_cases(patterns=["beliefs.identity", "reasoning.*"])[:2]
    src = asyncio.run(Runner(cfg, store).run(cases, "src"))
    src_results = {r["case_id"]: r for r in store.get_results(src)}

    new = asyncio.run(Runner(cfg, store).reannotate(src))
    assert new != src
    run = store.get_run(new)
    assert run["status"] == "finished" and run["config"]["reannotated_from"] == src
    for r in store.get_results(new):
        old = src_results[r["case_id"]]
        assert (r["status"], r["score"]) == (old["status"], old["score"])
        no_judge = lambda msgs: [(m["thread"], m["role"], m["content"]) for m in msgs if m["thread"] != "judge"]
        assert no_judge(store.get_messages(new, r["case_id"])) == no_judge(store.get_messages(src, r["case_id"]))
        if old["judge"]:
            assert r["judge"] and r["interest"] is not None
            # judge flags are replaced, not duplicated
            assert sum(f.get("source") == "judge" for f in r["flags"]) == len(r["judge"].get("flags") or [])
    assert store.get_results(src) == list(src_results.values())  # source untouched


def test_reannotate_case_subset(mock_url, tmp_path, monkeypatch):
    cfg = _cfg(monkeypatch, tmp_path, mock_url)
    store = Store(cfg.db_path)
    cases = select_cases(patterns=["beliefs.identity", "beliefs.lying"])
    src = asyncio.run(Runner(cfg, store).run(cases, "src"))
    new = asyncio.run(Runner(cfg, store).reannotate(src, ["beliefs.lying"]))
    assert [r["case_id"] for r in store.get_results(new)] == ["beliefs.lying"]


def test_listen_address(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    ns = argparse.Namespace
    assert _listen(cfg, ns(host=None, port=None))[:2] == ("127.0.0.1", 8765)
    host, port, url = _listen(cfg, ns(host="0.0.0.0", port=9000))
    assert (host, port) == ("0.0.0.0", 9000) and "all interfaces" in url
    (tmp_path / "tealeaf.toml").write_text('[web]\nhost = "0.0.0.0"\nport = 8800\n')
    assert _listen(load_config(), ns(host=None, port=None))[:2] == ("0.0.0.0", 8800)
