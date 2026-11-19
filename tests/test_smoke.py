"""End-to-end: run a slice of every suite against the mock server."""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from tealeaf.config import load_config
from tealeaf.mock_server import create_mock_app
from tealeaf.registry import SUITES, select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store


@pytest.fixture(scope="module")
def mock_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_mock_app(0), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True


def test_run_first_case_of_each_suite(mock_url, tmp_path, monkeypatch):
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
