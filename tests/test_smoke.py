"""End-to-end: run a slice of every suite against the mock server."""
import asyncio
import os

from tealeaf.config import load_config
from tealeaf.registry import SUITES, select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store


def test_run_first_case_of_each_suite(mock_url, tmp_path, monkeypatch):
    # Ignore the developer's own tealeaf.toml and TEALEAF_* env (e.g. a paid remote judge).
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_BASE_URL", mock_url)
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    cfg = load_config()
    cfg.run.samples, cfg.run.conversation_turns, cfg.run.long_context_max_tokens = 2, 2, 1000
    select_cases()  # register suites
    cases = [s.cases[0] for s in SUITES.values()] + select_cases(patterns=["conversations.self_talk"])
    store = Store(cfg.db_path)
    run_id = asyncio.run(Runner(cfg, store).run(cases, "smoke"))
    run = store.get_run(run_id)
    results = store.get_results(run_id)
    assert run["status"] == "finished"
    assert len(results) == len(cases)
    assert not [r for r in results if r["status"] == "error"], [(r["case_id"], r["error"]) for r in results]
    probes = [r for r in results if r["kind"] == "probe"]
    assert any(r["judge"] for r in probes)
    assert store.get_messages(run_id, cases[0].id)
    snap = run["config"]["model_prompt"]  # the model's prompt is snapshotted with the run
    assert snap["server"] == "ollama" and snap["model"] == run["target_model"]
