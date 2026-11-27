"""Unwrapped: identity with and without the chat template."""
import asyncio
import json
import os

import httpx
import pytest

from tealeaf.client import ChatClient, Reply
from tealeaf.config import Endpoint, load_config
from tealeaf.registry import select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store
from tealeaf.suites import unwrapped as U

from conftest import serve_mock


@pytest.fixture(scope="module")
def llamacpp_url():
    with serve_mock("llamacpp") as url:
        yield url


@pytest.fixture
def run_case(tmp_path, monkeypatch):
    def run(url, pattern="unwrapped.who"):
        monkeypatch.chdir(tmp_path)  # never pick up a real tealeaf.toml (and its judge)
        for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
            monkeypatch.delenv(k)
        monkeypatch.setenv("TEALEAF_BASE_URL", url)
        monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
        cfg = load_config()
        cfg.run.annotate = False
        store = Store(cfg.db_path)
        rid = asyncio.run(Runner(cfg, store).run(select_cases(patterns=[pattern])))
        return store, rid, store.get_result(rid, pattern)
    return run


def test_three_conditions_on_llamacpp(run_case, llamacpp_url):
    store, rid, r = run_case(llamacpp_url)
    ans = {a["id"]: a for a in r["metrics"]["answers"]}
    assert (ans["chat"]["identity"], ans["chat"]["name"]) == ("ai", "MockBot")
    assert ans["no_default_system"]["creator"] == "Alibaba Cloud"
    assert ans["bare"]["identity"] == "human" and "Q:" not in ans["bare"]["said"]  # the stop string ended it
    flags = {f["name"] for f in r["flags"]}
    assert flags == {"default_system_prompt_shapes_name", "default_system_prompt_shapes_creator",
                     "changes_without_template_identity", "changes_without_template_name"}
    assert r["score"] == 0.0  # a different kind of speaker answers without the template
    msgs = store.get_messages(rid, "unwrapped.who")
    bare = [m for m in msgs if m["thread"].startswith("bare")]
    assert bare[0]["content"] == "Q: Who are you?\nA:" and bare[0]["meta"]["raw_prompt"]
    nosys = [m for m in msgs if m["thread"].startswith("nosys")]
    assert nosys[0]["role"] == "system" and nosys[0]["meta"]["injected"]
    assert not any(m["meta"].get("injected") and m["role"] == "assistant" for m in msgs)  # no empty prefix turn


def test_chat_only_elsewhere(run_case, mock_url):
    _, _, r = run_case(mock_url)
    assert r["status"] == "observed" and r["score"] is None and "llama.cpp" in r["notes"][0]
    assert [a["id"] for a in r["metrics"]["answers"]] == ["chat"]


class Stub:
    """Enough of a Ctx for one unwrapped case, with a template that adds no system prompt."""

    def __init__(self):
        self.metrics, self.flags, self.notes, self.scores, self.threads = {}, [], [], [], 0
        outer = self

        class Target:
            async def raw_info(self):
                return {"ok": True, "template": "", "note": ""}

            async def render(self, messages):
                return f"<|user|>{messages[-1]['content']}<|assistant|>"

        class Judge:
            async def ask_json(self, instructions, content, purpose=""):
                return {"identity": "ai", "name": "Bot", "creator": "null"}

        self.target, self.judge = Target(), Judge()

    async def ask(self, q, **kw):
        return Reply(content="I'm Bot, an AI.")

    async def complete_text(self, text, thread, **kw):
        return Reply(content=" I'm Bot, an AI.")

    async def continue_raw(self, *a, **kw):
        raise AssertionError("no built-in system prompt: nothing to remove")

    def new_thread(self, prefix="t"):
        self.threads += 1
        return f"{prefix}{self.threads}"

    def record(self, *a, **kw):
        pass

    def metric(self, k, v):
        self.metrics[k] = v

    def flag(self, name, note=""):
        self.flags.append(name)

    def note(self, t):
        self.notes.append(t)

    def score(self, v):
        self.scores.append(v)


def test_template_without_a_built_in_system_prompt():
    ctx = Stub()
    asyncio.run(U._case("Who are you?")(ctx))
    assert [a["id"] for a in ctx.metrics["answers"]] == ["chat", "bare"]
    assert any("no built-in system prompt" in n for n in ctx.notes)
    assert ctx.scores == [1.0] and not ctx.flags
    assert ctx.metrics["answers"][0]["creator"] is None  # "null" from the judge means none


def test_helpers():
    assert U._same("OpenAI", "openai.") and not U._same("OpenAI", "Anthropic")
    assert U._s(" Qwen ") == "Qwen" and U._s("None") is None and U._s(None) is None


def test_raw_completion_sends_stop_strings():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"content": " hi", "tokens_predicted": 1, "stop_type": "word"})

    async def go():
        c = ChatClient(Endpoint(base_url="http://mock/v1", model="m"))
        c._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.complete_raw("Q: x\nA:", stop=["\nQ:"])
        await c.complete_raw("no stop")
        await c.aclose()

    asyncio.run(go())
    assert sent[0]["stop"] == ["\nQ:"] and "stop" not in sent[1]
