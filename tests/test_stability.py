"""Run-level system prompt, repeat groups, and stability across repeated runs."""
import asyncio
import os

import pytest

from tealeaf.config import load_config
from tealeaf.registry import select_cases
from tealeaf.runner import Runner
from tealeaf.stability import compute, judge_consistency, new_group_id, resolve_runs
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


def _run(cfg, store, cases, meta=None):
    async def go():
        r = Runner(cfg, store)
        await r.prepare(cases, "t", meta=meta)
        return await r.run(cases, "t")
    return asyncio.run(go())


def test_run_system_prompt_reaches_target_only(setup):
    cfg, store = setup
    cfg.run.system_prompt = "RUNPROMPT: you work for Acme."
    cases = select_cases(patterns=["beliefs.identity", "conversations.persona_persistence"])
    rid = _run(cfg, store, cases)
    assert store.get_run(rid)["config"]["run"]["system_prompt"] == cfg.run.system_prompt
    for c in cases:
        msgs = store.get_messages(rid, c.id)
        systems = [m for m in msgs if m["role"] == "system"]
        assert systems and all(m["content"].startswith("RUNPROMPT") and m["meta"]["run_system_prompt"]
                               for m in systems)
        assert not any("RUNPROMPT" in m["content"] for m in msgs if m["thread"] == "judge")
    persona = [m for m in store.get_messages(rid, "conversations.persona_persistence") if m["role"] == "system"]
    assert "Captain Mara" in persona[0]["content"]  # the case's own system prompt follows the run's


def test_repeat_group_and_stability(setup):
    cfg, store = setup
    cases = select_cases(patterns=["beliefs.identity"]) + select_cases(patterns=["reasoning.*"])[:1]
    gid = new_group_id()
    ids = [_run(cfg, store, cases, {"group": {"id": gid, "index": i + 1, "of": 3}}) for i in range(3)]

    g = store.groups()[0]
    assert g["id"] == gid and g["runs"] == ids
    assert resolve_runs(store, [gid]) == (gid, ids)
    assert resolve_runs(store, [ids[1][:-1], ids[0]])[1] == sorted(ids[:2])  # prefixes ok
    with pytest.raises(ValueError):
        resolve_runs(store, [ids[0]])

    rep = compute(store, gid, ids)
    by = {c["case_id"]: c for c in rep["cases"]}
    assert set(by) == {c.id for c in cases} and not rep["warnings"]
    probe = by["beliefs.identity"]
    trad = next(c for c in rep["cases"] if c["kind"] == "traditional")
    assert probe["stability"] is None and probe["n"] == 3  # probes need the judge
    assert 0 <= trad["stability"] <= 1 and trad["n"] == 3

    asyncio.run(judge_consistency(store, gid, ids, cfg.judge_endpoint()))
    rep = compute(store, gid, ids)  # judged result is cached and merged
    probe = next(c for c in rep["cases"] if c["case_id"] == "beliefs.identity")
    assert rep["judge_model"] and 0.3 <= probe["stability"] <= 1 and probe["judge"]["unstable"]
