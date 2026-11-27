"""Dashboard request guard, API keys staying with their server, run crash status, client retries, code sandbox."""
import asyncio
import json
import os
import sys
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from tealeaf.client import ChatClient, ClientError
from tealeaf.config import Endpoint, derive_judge, load_config
from tealeaf.registry import Case, select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store
from tealeaf.suites.coding import run_tests
from tealeaf.web.app import create_app

from conftest import serve_mock

KEY = "sk-hardening"


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    return monkeypatch


@pytest.fixture(scope="module")
def keyed_url():
    with serve_mock(flavor="openai", api_key=KEY) as url:
        yield url


# --------------------------------------------------------------------------- dashboard guard
def _client(cfg, **kw):
    return TestClient(create_app(cfg, Store(cfg.db_path)), **kw)


def test_unknown_host_is_refused(clean_env):
    cfg = load_config()
    c = _client(cfg, base_url="http://rebind.attacker.example")
    assert c.get("/api/runs").status_code == 403
    for ok in ("http://localhost:8765", "http://127.0.0.1:8765", "http://[::1]:8765"):
        assert _client(cfg, base_url=ok).get("/api/runs").status_code == 200
    cfg.web.allowed_hosts = ["tealeaf.lan"]
    assert _client(cfg, base_url="http://tealeaf.lan").get("/api/runs").status_code == 200


def test_state_changes_need_the_page_header(clean_env):
    c = _client(load_config(), base_url="http://localhost")
    assert c.post("/api/runs/x/cancel").status_code == 403
    assert c.post("/api/runs/x/resume").status_code == 403
    assert c.delete("/api/runs/x").status_code == 403
    # A URL from the request makes the server call out (with a key, possibly): treated as a state change.
    assert c.get("/api/models", params={"base_url": "http://evil.example/v1"}).status_code == 403
    assert c.post("/api/runs/x/cancel", headers={"X-Tealeaf": "1"}).status_code == 200
    assert c.get("/api/runs").status_code == 200  # plain reads are fine


def test_cross_origin_requests_are_refused(clean_env):
    c = _client(load_config(), base_url="http://localhost:8765", headers={"X-Tealeaf": "1"})
    assert c.post("/api/runs/x/cancel", headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/runs/x/cancel", headers={"Origin": "http://localhost:8765"}).status_code == 200


def test_configured_key_is_not_sent_to_a_typed_in_url(clean_env, keyed_url):
    clean_env.setenv("TEALEAF_BASE_URL", keyed_url)
    clean_env.setenv("TEALEAF_API_KEY", KEY)
    c = _client(load_config(), base_url="http://localhost", headers={"X-Tealeaf": "1"})
    assert c.get("/api/models").json()["error"] is None  # configured server: configured key
    # Same mock server under another name stands in for "some other host": it must not get the key.
    other = keyed_url.replace("127.0.0.1", "localhost")
    assert c.get("/api/models", params={"base_url": other}).json()["error"]
    assert c.get("/api/models", params={"base_url": other}, headers={"X-Api-Key": KEY}).json()["error"] is None


# --------------------------------------------------------------------------- keys stay with their server
def test_config_judge_on_another_server_does_not_inherit_target_key(clean_env, tmp_path):
    (tmp_path / "tealeaf.toml").write_text(
        '[target]\nbase_url = "https://hosted/v1"\napi_key = "target-key"\n'
        '[judge]\nbase_url = "http://localhost:11434/v1"\nmodel = "j"\n')
    assert load_config().judge.api_key == ""
    (tmp_path / "tealeaf.toml").write_text(
        '[target]\nbase_url = "https://hosted/v1"\napi_key = "target-key"\n[judge]\nmodel = "j"\n')
    assert load_config().judge.api_key == "target-key"  # same server


def test_derive_judge_keeps_key_only_for_same_server(clean_env):
    t = Endpoint(base_url="https://hosted/v1", api_key="k")
    assert derive_judge(t).api_key == "k"
    assert derive_judge(t, "http://localhost:1234/v1").api_key == ""
    clean_env.setenv("TEALEAF_JUDGE_BASE_URL", "http://elsewhere/v1")
    assert derive_judge(t).api_key == ""  # env moved it to another server


# --------------------------------------------------------------------------- runner
@pytest.fixture
def mock_cfg(clean_env, mock_url):
    clean_env.setenv("TEALEAF_BASE_URL", mock_url)
    cfg = load_config()
    cfg.run.samples, cfg.run.conversation_turns = 2, 2
    return cfg


def test_crashed_run_is_marked_error(mock_cfg, monkeypatch):
    store = Store(mock_cfg.db_path)
    runner = Runner(mock_cfg, store)

    async def boom(case):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(runner, "_run_case", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(runner.run(select_cases(patterns=["reasoning.*"])[:1], "t"))
    run = store.get_run(runner.run_id)
    assert run["status"] == "error" and "db exploded" in run["error"]


def test_judge_calls_are_not_counted_as_target_usage(mock_cfg):
    async def fn(ctx):
        await ctx.ask("hello")
        await ctx.judge.verdict("Is it polite?", "hello", ["yes", "no"])
        await ctx.interlocutor("A curious user.").next(None)

    store = Store(mock_cfg.db_path)
    case = Case(suite="t", name="usage", title="usage", fn=fn, kind="probe")
    rid = asyncio.run(Runner(mock_cfg, store).run([case]))
    assert mock_cfg.judge is None  # the target judges itself, so judge calls go through the target client
    assert store.get_result(rid, case.id)["metrics"]["calls"] == 1


def test_client_does_not_modify_the_shared_endpoint(mock_url):
    ep = Endpoint(base_url=mock_url, model="")
    client = ChatClient(ep)
    assert asyncio.run(client.resolve_model())
    assert ep.model == ""


# --------------------------------------------------------------------------- client retries
SSE = 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'


def _client_with(handler) -> ChatClient:
    c = ChatClient(Endpoint(base_url="http://mock/v1", model="m"))
    c._http = httpx.AsyncClient(base_url="http://mock/v1", transport=httpx.MockTransport(handler))
    return c


def test_400_disables_stream_options_only_when_that_was_the_cause():
    def picky(req):
        if "stream_options" in json.loads(req.content):
            return httpx.Response(400, text="unknown field stream_options")
        return httpx.Response(200, text=SSE)

    c = _client_with(picky)
    assert asyncio.run(c.chat([{"role": "user", "content": "x"}])).content == "hi"
    assert c._stream_options_ok is False

    c = _client_with(lambda req: httpx.Response(400, text="context length exceeded"))
    with pytest.raises(ClientError, match="context length"):
        asyncio.run(c.chat([{"role": "user", "content": "x"}]))
    assert c._stream_options_ok is True  # a real 400 doesn't switch off usage reporting


def test_rate_limits_and_server_errors_are_retried():
    seen = []

    def flaky(req):
        seen.append(1)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if len(seen) == 2:
            return httpx.Response(503, headers={"Retry-After": "0"}, text="model loading")
        return httpx.Response(200, text=SSE)

    assert asyncio.run(_client_with(flaky).chat([{"role": "user", "content": "x"}])).content == "hi"
    assert len(seen) == 3


# --------------------------------------------------------------------------- code sandbox
def test_model_code_cannot_start_processes():
    out = asyncio.run(run_tests("import subprocess\nsubprocess.run(['true'])\ndef f(): return 1", ["f() == 1"]))
    assert "error" in out


def test_model_code_timeout_kills_it():
    assert asyncio.run(run_tests("while True: pass", ["True"], timeout=1)) == {"error": "timeout"}


@pytest.mark.skipif(sys.platform != "darwin", reason="filesystem/network isolation uses macOS sandbox-exec")
def test_model_code_cannot_write_outside_its_folder_or_use_network(tmp_path):
    target = tmp_path / "escaped.txt"
    out = asyncio.run(run_tests(f"open({str(target)!r}, 'w').write('x')\ndef f(): return 1", ["f() == 1"]))
    assert "error" in out and not target.exists()
    out = asyncio.run(run_tests("import socket\nsocket.create_connection(('1.1.1.1', 53), 2)\ndef f(): return 1",
                                ["f() == 1"]))
    assert "error" in out
