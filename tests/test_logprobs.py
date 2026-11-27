"""Token log-probabilities: parsing, analysis, client capture, storage, and the probes that use them."""
import asyncio
import json
import math
import os
import sqlite3

import httpx
import pytest

from tealeaf import logprobs as lp
from tealeaf.client import ChatClient
from tealeaf.config import Endpoint, load_config
from tealeaf.registry import select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store


def entry(tok, p, alts=(), reasoning=False):
    return [tok, round(math.log(p), 4), [[t, round(math.log(q), 4)] for t, q in alts], reasoning]


def api_items(*toks):
    """OpenAI-style logprobs content for (token, p, [(alt, q), ...]) tuples."""
    return [{"token": t, "logprob": math.log(p), "top_logprobs": [{"token": a, "logprob": math.log(q)} for a, q in alts]}
            for t, p, alts in toks]


# --------------------------------------------------------------------------- parsing and analysis
def test_parse_and_mark_inline_thinking():
    toks = lp.finalize(lp.parse(api_items(("<think>", .9, [("<think>", .9)]), ("hmm", .5, [("ok", .3)]),
                                          ("</think>", .9, []), ("B", .7, [("B", .7), ("C", .2)]))))
    assert [t[3] for t in toks] == [True, True, True, False]
    assert toks[1][2] == [] and toks[3][2]  # reasoning keeps no alternatives; the answer does
    # Template injected the opening <think>: everything up to </think> is reasoning.
    toks = lp.finalize(lp.parse(api_items(("plan", .5, []), ("</think>", .9, []), ("Yes", .8, []))))
    assert [t[3] for t in toks] == [True, True, False]


def test_parse_skips_junk_and_infinities():
    toks = lp.parse([{"token": "a", "logprob": float("-inf"), "top_logprobs": []}, {"logprob": -1}, "x"])
    assert toks == [["a", -99.0, [], False]]


def test_option_probs_reads_the_first_meaningful_token():
    toks = [entry("**", .99), entry("B", .6, [("B", .6), ("C", .3), ("A", .05), ("Hello", .05)]), entry("**", .99)]
    p = lp.option_probs(toks, list("ABCD"))
    assert p["B"] == pytest.approx(.6 / .95, abs=1e-3) and p["C"] == pytest.approx(.3 / .95, abs=1e-3)
    assert p["_mass"] == pytest.approx(.95, abs=1e-3)
    # " Yes" and "Yes" both count for Yes; prefixes ("Ye") too.
    p = lp.option_probs([entry(" Yes", .5, [(" Yes", .5), ("Yes", .2), ("No", .2), ("Ye", .1)])], ["Yes", "No"])
    assert p["Yes"] == pytest.approx(.8) and p["No"] == pytest.approx(.2)
    assert lp.option_probs(None, ["Yes"]) is None
    assert lp.option_probs([entry("x", .9, [], reasoning=True)], ["Yes"]) is None  # nothing visible


def test_alternatives_merge_whitespace_variants():
    alts = lp.alternatives([entry("Open", .5, [("Open", .5), (" Open", .1), ("Ali", .3)])])
    assert alts[0] == ("Open", pytest.approx(.6, abs=1e-3)) and alts[1][0] == "Ali"


def test_stats_counts_forks():
    toks = [entry("The", .95, [("The", .95)]), entry(" cat", .5, [(" cat", .5), (" dog", .4)]), entry(".", .9)]
    st = lp.stats(toks)
    assert st["tokens"] == 3 and st["forks"] == 1 and st["min_p"] == pytest.approx(.5)
    assert st["perplexity"] == pytest.approx(math.exp(-(math.log(.95) + math.log(.5) + math.log(.9)) / 3), abs=1e-3)


def test_calibration():
    assert lp.calibration([(.9, True)]) is None  # too few
    c = lp.calibration([(.9, True), (.9, False), (.9, True), (.9, False)])
    assert c["accuracy"] == .5 and c["mean_confidence"] == .9 and c["overconfidence"] == .4 and c["ece"] == .4
    assert lp.calibration([(1.0, True)] * 5)["ece"] == 0


# --------------------------------------------------------------------------- client
def _client(handler, stream=True) -> ChatClient:
    c = ChatClient(Endpoint(base_url="http://mock/v1", model="m", stream=stream))
    c._http = httpx.AsyncClient(base_url="http://mock/v1", transport=httpx.MockTransport(handler))
    return c


def _sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_stream_captures_logprobs_and_reasoning_tokens():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, text=_sse([
            {"choices": [{"delta": {"reasoning_content": "think"},
                          "logprobs": {"content": api_items(("think", .4, [("think", .4), ("ponder", .3)]))}}]},
            {"choices": [{"delta": {"content": "B"}, "finish_reason": "stop",
                          "logprobs": {"content": api_items(("B", .7, [("B", .7), ("C", .2)]))}}]}]))

    r = asyncio.run(_client(handler).chat([{"role": "user", "content": "q"}]))
    assert sent[0]["logprobs"] is True and sent[0]["top_logprobs"] == 5
    assert [t[3] for t in r.logprobs] == [True, False]
    assert r.option_probs(list("ABCD"))["B"] == pytest.approx(.7 / .9, abs=1e-3)
    assert r.stats()["logprob_stats"]["tokens"] == 1


def test_non_stream_captures_logprobs():
    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "No"}, "finish_reason": "stop",
                                                      "logprobs": {"content": api_items(("No", .6, [("Yes", .4)]))}}]})

    r = asyncio.run(_client(handler, stream=False).chat([{"role": "user", "content": "q"}]))
    assert r.option_probs(["Yes", "No"]) == {"Yes": pytest.approx(.4), "No": pytest.approx(.6), "_mass": 1.0}


def test_logprobs_false_is_not_requested():
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, text=_sse([{"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}]))

    r = asyncio.run(_client(handler).chat([{"role": "user", "content": "q"}], logprobs=False))
    assert "logprobs" not in sent[0] and r.logprobs is None


@pytest.mark.parametrize("rejected", ["stream_options", "logprobs"])
def test_a_rejected_field_is_dropped_alone(rejected):
    def handler(req):
        if rejected in json.loads(req.content):
            return httpx.Response(400, text=f"unsupported: {rejected}")
        return httpx.Response(200, text=_sse([{"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}]))

    c = _client(handler)
    asyncio.run(c.chat([{"role": "user", "content": "q"}]))
    assert c._off == {rejected}  # the other one keeps working


# --------------------------------------------------------------------------- storage
def test_old_database_gets_the_logprobs_column(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, case_id TEXT, "
                "thread TEXT, role TEXT, content TEXT, reasoning TEXT, meta TEXT, ts REAL)")
    con.execute("INSERT INTO messages (run_id, case_id, content) VALUES ('r', 'c', 'old')")
    con.commit()
    con.close()
    st = Store(str(db))
    st.add_message("r", "c", "t1", "assistant", "new", logprobs=[entry("new", .5)])
    old, new = st.get_messages("r", "c")
    assert old["logprobs"] is None and new["logprobs"][0][0] == "new"


# --------------------------------------------------------------------------- end to end with the mock server
@pytest.fixture
def mock_cfg(mock_url, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("TEALEAF_BASE_URL", mock_url)
    monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
    cfg = load_config()
    cfg.run.annotate = False
    return cfg


def test_latent_and_calibration_end_to_end(mock_cfg):
    store = Store(mock_cfg.db_path)
    events = []
    runner = Runner(mock_cfg, store, listeners=[lambda t, p: events.append((t, p))])
    rid = asyncio.run(runner.run(select_cases(["latent", "knowledge"]), "lp"))
    run = store.get_run(rid)
    assert run["summary"]["calibration"]["n"] >= 3
    res = {r["case_id"]: r for r in store.get_results(rid)}
    assert res["latent.capital"]["metrics"]["outcome"] in ("held", "caved_against_belief", "persuaded")
    assert 0 <= res["latent.capital"]["score"] <= 1
    assert res["latent.belief_coherence"]["metrics"]["beliefs"]
    assert res["latent.identity_alternatives"]["metrics"]["creator"]["alternatives"]
    assert "confidence" in res["knowledge.tungsten"]["metrics"]
    msgs = [m for m in store.get_messages(rid, "latent.capital") if m["role"] == "assistant"]
    assert msgs and all(m["logprobs"] for m in msgs) and msgs[0]["meta"]["logprob_stats"]
    # Live events stay light: no token data, just a marker.
    ev = [p for t, p in events if t == "message" and p["role"] == "assistant"]
    assert ev and all(p["logprobs"] is None and p["has_logprobs"] for p in ev)


def test_cases_without_logprobs_stay_unscored(mock_cfg):
    mock_cfg.target.logprobs = False
    store = Store(mock_cfg.db_path)
    rid = asyncio.run(Runner(mock_cfg, store).run(select_cases(patterns=["latent.capital", "knowledge.tungsten"])))
    res = {r["case_id"]: r for r in store.get_results(rid)}
    assert res["latent.capital"]["status"] == "observed" and res["latent.capital"]["score"] is None
    assert "confidence" not in res["knowledge.tungsten"]["metrics"]
    assert store.get_run(rid)["summary"]["calibration"] is None


def test_judge_calls_do_not_request_logprobs(mock_cfg):
    mock_cfg.run.annotate = True
    store = Store(mock_cfg.db_path)
    rid = asyncio.run(Runner(mock_cfg, store).run(select_cases(patterns=["beliefs.*"])[:1]))
    judged = [m for r in store.get_results(rid) for m in store.get_messages(rid, r["case_id"]) if m["thread"] == "judge"]
    assert judged and not any(m["logprobs"] for m in judged)
