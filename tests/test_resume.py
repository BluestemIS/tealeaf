"""Continuing a stopped run in place."""
import asyncio
import os
from dataclasses import replace

import pytest

from tealeaf.config import Endpoint, load_config, restore_config
from tealeaf.registry import select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store


@pytest.fixture
def setup(mock_url, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_BASE_URL", mock_url)
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    cfg = load_config()
    cfg.run.samples, cfg.run.conversation_turns = 2, 2
    return cfg, Store(cfg.db_path)


def test_resume_reruns_unfinished_cases_with_original_settings(setup):
    cfg, store = setup
    cfg.run.system_prompt = "ORIGINAL PROMPT"
    cases = select_cases(patterns=["beliefs.identity", "beliefs.lying"]) + select_cases(patterns=["reasoning.*"])[:2]
    rid = asyncio.run(Runner(cfg, store).run(cases, "t"))
    kept, skipped, errored = cases[0].id, cases[1].id, cases[2].id
    before = store.get_result(rid, kept)
    # simulate a run that died: one case never ran, one errored
    store.clear_result(rid, skipped)
    store.update_result(rid, skipped, status="skipped")
    store.add_message(rid, skipped, "t1", "user", "partial leftover", "", None)
    store.update_result(rid, errored, status="error", error="timed out")
    store.update_run(rid, status="interrupted", done_cases=2)

    now = replace(cfg, run=replace(cfg.run, system_prompt="SOMETHING ELSE"))  # today's config differs
    restored = restore_config(now, store.get_run(rid)["config"])
    assert restored.run.system_prompt == "ORIGINAL PROMPT"

    _, n = asyncio.run(Runner(restored, store).resume(rid))
    assert n == 1
    run = store.get_run(rid)
    assert run["status"] == "finished" and run["done_cases"] == 4  # errored cases count as done, as in a normal run
    assert store.get_result(rid, kept)["finished_at"] == before["finished_at"]  # untouched
    assert store.get_result(rid, skipped)["status"] == "observed"
    msgs = store.get_messages(rid, skipped)
    assert not any(m["content"] == "partial leftover" for m in msgs)
    assert any(m["role"] == "system" and m["content"].startswith("ORIGINAL PROMPT") for m in msgs)
    assert store.get_result(rid, errored)["status"] == "error"  # errors only with include_errors

    store.update_run(rid, status="finished")
    _, n = asyncio.run(Runner(restored, store).resume(rid, include_errors=True))
    assert n == 1 and store.get_result(rid, errored)["status"] in ("pass", "fail")
    assert store.get_run(rid)["done_cases"] == 4


def test_resume_refuses_active_runs(setup):
    cfg, store = setup
    cases = select_cases(patterns=["reasoning.*"])[:1]
    rid = asyncio.run(Runner(cfg, store).run(cases, "t"))
    store.update_run(rid, status="running")
    with pytest.raises(ValueError, match="still running"):
        asyncio.run(Runner(cfg, store).resume(rid))


def test_restore_config_keys_come_from_current_config(monkeypatch):
    monkeypatch.setenv("TEALEAF_JUDGE_API_KEY", "judge-key")
    from tealeaf.config import Config, OpikSettings, RunSettings
    now = Config(target=Endpoint(api_key="target-key"), judge=None, run=RunSettings(), opik=OpikSettings(),
                 db_path="x")
    stored = {"target": {"base_url": "http://a/v1", "model": "m", "api_key": "***", "max_tokens": 123},
              "judge": {"base_url": "https://openrouter.ai/api/v1", "model": "j", "api_key": "***"},
              "run": {"samples": 3, "unknown_future_setting": 1}}
    c = restore_config(now, stored)
    # The run used another server than today's target, so today's key doesn't go along.
    assert (c.target.base_url, c.target.model, c.target.max_tokens, c.target.api_key) == ("http://a/v1", "m", 123, "")
    now.target.base_url = "http://a/v1/"
    assert restore_config(now, stored).target.api_key == "target-key"  # same server -> its key
    assert (c.judge.model, c.judge.api_key) == ("j", "judge-key")
    assert c.run.samples == 3
    assert restore_config(now, {**stored, "judge": None}).judge is None  # it judged itself
