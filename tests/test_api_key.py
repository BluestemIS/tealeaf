"""The target's API key: from env, from a named env var (--api-key-env), or typed into the dashboard."""
import argparse
import os
import time

import pytest
from fastapi.testclient import TestClient

from tealeaf.cli import _apply_overrides
from tealeaf.config import load_config
from tealeaf.storage import Store
from tealeaf.web.app import create_app

from conftest import serve_mock

KEY = "sk-test-123"


@pytest.fixture(scope="module")
def keyed_url():
    with serve_mock(flavor="openai", api_key=KEY) as url:
        yield url


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    return monkeypatch


def ns(**kw):
    return argparse.Namespace(**kw)


def test_env_var_sets_target_key(clean_env):
    clean_env.setenv("TEALEAF_API_KEY", KEY)
    assert load_config().target.api_key == KEY


def test_api_key_env_reads_named_variable(clean_env):
    clean_env.setenv("MY_PROVIDER_KEY", KEY)
    cfg = _apply_overrides(load_config(), ns(api_key_env="MY_PROVIDER_KEY"))
    assert cfg.target.api_key == KEY


def test_api_key_env_missing_variable_exits(clean_env):
    clean_env.delenv("NOPE_NOT_SET", raising=False)
    with pytest.raises(SystemExit):
        _apply_overrides(load_config(), ns(api_key_env="NOPE_NOT_SET"))


def test_api_key_env_not_sent_to_a_different_judge_server(clean_env):
    clean_env.setenv("MY_PROVIDER_KEY", KEY)
    cfg = _apply_overrides(load_config(), ns(api_key_env="MY_PROVIDER_KEY", base_url="https://hosted/v1",
                                             judge_url="http://localhost:11434/v1", judge_model="j"))
    assert cfg.target.api_key == KEY and cfg.judge.api_key == ""
    cfg = _apply_overrides(load_config(), ns(api_key_env="MY_PROVIDER_KEY", base_url="https://hosted/v1",
                                             judge_url=None, judge_model="j"))
    assert cfg.judge.api_key == KEY  # same server -> same key


def test_dashboard_key_lists_models_and_runs(clean_env, keyed_url):
    clean_env.setenv("TEALEAF_BASE_URL", keyed_url)
    cfg = load_config()
    store = Store(cfg.db_path)
    with TestClient(create_app(cfg, store), base_url="http://localhost", headers={"X-Tealeaf": "1"}) as c:  # context manager keeps the app's loop (and the run) alive
        _check_dashboard(c, store)


def _check_dashboard(c, store):
    assert c.get("/api/models").json()["error"]  # no key -> 401 from the server
    d = c.get("/api/models", headers={"X-Api-Key": KEY}).json()
    assert d["error"] is None and d["models"]

    r = c.post("/api/runs", json={"cases": ["reasoning.*"], "models": [d["models"][0]["id"]], "api_key": KEY})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    for _ in range(100):
        run = store.get_run(rid)
        if run["status"] not in ("running", "queued"):
            break
        time.sleep(0.1)
    assert run["status"] == "finished"
    assert run["config"]["target"]["api_key"] == "***"  # the key itself is never stored
    assert KEY not in str(run["config"])
    assert not [x for x in store.get_results(rid) if x["status"] == "error"]
