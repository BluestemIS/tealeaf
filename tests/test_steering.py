"""Steering: the agent (with fake llama.cpp binaries), a full sweep, the report, CLI and dashboard API."""
import asyncio
import contextlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from tealeaf.config import load_config
from tealeaf.steer_agent import Agent, AgentFailure, create_agent_app
from tealeaf.steering import CONCEPTS, AgentClient, AgentError, parse_scales, prompt_pairs, report, sweep
from tealeaf.storage import Store
from tealeaf.web.app import create_app

TOKEN = "steer-test-token"

FAKE_SERVER = '''#!{py}
import json, os, sys
args = sys.argv[1:]
if "--help" in args:
    print("{help}")
    sys.exit(0)
def opt(name, n=1):
    i = args.index(name)
    return args[i + 1:i + 1 + n]
scale = 0.0
if "--control-vector-scaled" in args:
    path, s = opt("--control-vector-scaled", 2)
    assert open(path).read() == "fake vector"
    scale = float(s)
with open(os.environ["FAKE_LLAMA_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
os.environ["TEALEAF_MOCK_STEER"] = str(scale)
import uvicorn
from tealeaf.mock_server import create_mock_app
uvicorn.run(create_mock_app(0, "llamacpp"), host="127.0.0.1", port=int(opt("--port")[0]), log_level="error")
'''

FAKE_GENERATOR = '''#!{py}
import sys
args = sys.argv[1:]
if "--help" in args:
    print("--positive-file FNAME --negative-file FNAME --method pca|mean -o FNAME")
    sys.exit(0)
def opt(name):
    return args[args.index(name) + 1]
pos = open(opt("--positive-file")).read().splitlines()
neg = open(opt("--negative-file")).read().splitlines()
assert len(pos) == len(neg) and pos != neg and all(pos)
open(opt("-o"), "w").write("fake vector")
'''


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _script(path, text):
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def fake_llama(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "llama-server", FAKE_SERVER.format(
        py=sys.executable, help="--control-vector-scaled FNAME SCALE --control-vector-layer-range START END"))
    _script(bin_dir / "llama-cvector-generator", FAKE_GENERATOR.format(py=sys.executable))
    models = tmp_path / "models"
    models.mkdir()
    (models / "mock.gguf").write_text("weights")
    log = tmp_path / "llama.log"
    monkeypatch.setenv("FAKE_LLAMA_LOG", str(log))
    return {"bin": str(bin_dir), "models": str(models), "data": str(tmp_path / "steer"), "log": log}


def make_agent(fake, **kw) -> Agent:
    return Agent(bin_dir=fake["bin"], models_dir=fake["models"], default_model="mock.gguf", data_dir=fake["data"],
                 server_host="127.0.0.1", server_port=_port(), health_timeout=60, **kw)


@contextlib.contextmanager
def serve_agent(agent: Agent):
    port = _port()
    server = uvicorn.Server(uvicorn.Config(create_agent_app(agent, TOKEN), host="127.0.0.1", port=port,
                                           log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        agent.stop()


def launches(fake) -> list[list[str]]:
    return [json.loads(x) for x in fake["log"].read_text().splitlines()] if fake["log"].exists() else []


# --------------------------------------------------------------------------- agent
def test_agent_checks_llama_cpp_flags(fake_llama):
    _script(Path(fake_llama["bin"]) / "llama-server", FAKE_SERVER.format(py=sys.executable, help="--port PORT"))
    with pytest.raises(AgentFailure, match="control-vector"):
        make_agent(fake_llama)


def test_agent_needs_token_and_stays_in_its_models_folder(fake_llama):
    c = TestClient(create_agent_app(make_agent(fake_llama), TOKEN))
    assert c.get("/status").status_code == 401
    assert c.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    auth = {"Authorization": f"Bearer {TOKEN}"}
    st = c.get("/status", headers=auth).json()
    assert st["models"] == ["mock.gguf"] and st["vectors"] == [] and not st["server"]["running"]
    r = c.post("/server", headers=auth, json={"model": "../../etc/passwd.gguf"})
    assert r.status_code == 400 and "models folder" in r.json()["detail"]
    r = c.post("/server", headers=auth, json={"vector": "honesty", "scale": 1.0})
    assert r.status_code == 400 and "build it first" in r.json()["detail"]
    with pytest.raises(AgentFailure, match="names"):
        make_agent(fake_llama).vector_path(Path("m.gguf"), "../evil")


# --------------------------------------------------------------------------- full sweep
@pytest.fixture
def cfg(mock_url, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_BASE_URL", mock_url)  # the normal, unsteered server: it judges
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    return load_config()


def test_emotion_sweep_end_to_end(fake_llama, cfg, mock_url):
    agent = make_agent(fake_llama)
    store = Store(cfg.db_path)
    with serve_agent(agent) as url:
        async def go():
            client = AgentClient(url, TOKEN)
            try:
                return await sweep(cfg, store, client, CONCEPTS["emotion"], [-1.0, 0.0, 1.0])
            finally:
                await client.aclose()

        sid = asyncio.run(go())
    # The vector was built from rendered, escaped prompt pairs.
    built = list((agent.data_dir / "vectors" / "mock").glob("emotion.gguf"))
    assert built and built[0].read_text() == "fake vector"
    pos = (agent.data_dir / "build" / "mock" / "emotion.positive.txt").read_text().splitlines()
    assert len(pos) == len(prompt_pairs(CONCEPTS["emotion"])) and "extremely happy" in pos[0] and "\\n" in pos[0]
    # One plain start to render the prompts, then one per strength; 0 runs without a vector.
    runs = launches(fake_llama)
    assert len(runs) == 4
    vec = [r[r.index("--control-vector-scaled") + 2] if "--control-vector-scaled" in r else None for r in runs]
    assert vec == [None, "-1.0", None, "1.0"]
    assert agent.proc is None  # stopped afterwards

    rep = report(store, sid)
    assert [s["scale"] for s in rep["steps"]] == [-1.0, 0.0, 1.0]
    tone = next(c for c in rep["cases"] if c["case_id"] == "mood.tone")
    assert [tone["by_scale"][k]["score"] for k in ("-1", "0", "1")] == [0.0, 0.5, 1.0]
    assert tone["range"] == 1.0 and tone["slope"] > 0
    happy = next(c for c in rep["cases"] if c["case_id"] == "mood.feels_happy")
    assert happy["by_scale"]["-1"]["score"] == 0.0 and happy["by_scale"]["1"]["score"] == 1.0
    assert rep["suites"]["mood"]["1"] > rep["suites"]["mood"]["-1"] and "0" in rep["suites"]["mood"]
    # The steered model never graded itself.
    for s in rep["steps"]:
        run = store.get_run(s["run_id"])
        assert run["config"]["judge"]["base_url"] == mock_url != run["target_url"]

    # The sweep shows up in the CLI and the dashboard.
    from tealeaf import cli
    cli.print_steering(rep)
    with TestClient(create_app(cfg, store), base_url="http://localhost") as c:
        assert c.get("/api/steering").json()[0]["id"] == sid
        assert c.get(f"/api/steering/{sid}").json()["concept"] == "emotion"
        assert c.get("/api/steering/nope").status_code == 404


def test_sweep_needs_a_reachable_judge(fake_llama, cfg):
    cfg.target.base_url = f"http://127.0.0.1:{_port()}/v1"  # nothing listening
    agent = make_agent(fake_llama)
    with serve_agent(agent) as url:
        async def go():
            client = AgentClient(url, TOKEN)
            try:
                await sweep(cfg, Store(cfg.db_path), client, CONCEPTS["emotion"], [0.0, 1.0])
            finally:
                await client.aclose()

        with pytest.raises(AgentError, match="judge must stay reachable"):
            asyncio.run(go())


def test_agent_client_errors_are_readable():
    async def go():
        c = AgentClient(f"http://127.0.0.1:{_port()}", TOKEN)
        try:
            await c.status()
        finally:
            await c.aclose()

    with pytest.raises(AgentError, match="steer agent"):
        asyncio.run(go())
    with pytest.raises(AgentError, match="agent_url"):
        AgentClient("", TOKEN)


def test_parse_scales_and_concepts():
    assert parse_scales("1, -1,0,0") == [-1.0, 0.0, 1.0]
    assert parse_scales([0.5, "-0.5"]) == [-0.5, 0.5]
    with pytest.raises(ValueError):
        parse_scales("")
    from tealeaf.registry import select_cases
    for c in CONCEPTS.values():  # every concept measures cases that exist
        for pat in c.cases:
            assert select_cases(patterns=[pat]), (c.name, pat)
        pairs = prompt_pairs(c)
        assert all(p["positive"][1] == p["negative"][1] for p in pairs)  # only the persona differs
