"""Contamination: splitting and comparing text, loading benchmark files, and the suite end to end."""
import argparse
import asyncio
import json
import os
import re

import pytest

from tealeaf.cli import _apply_overrides
from tealeaf.config import load_config
from tealeaf.registry import select_cases
from tealeaf.runner import Runner
from tealeaf.storage import Store
from tealeaf.suites import contamination as C

from conftest import serve_mock


@pytest.fixture(scope="module")
def llamacpp_url():
    with serve_mock("llamacpp") as url:
        yield url


def test_split_keeps_text_and_never_ends_the_prompt_with_a_space():
    for _, text in C.KNOWN + C.NOVEL + C.BENCHMARKS:
        prefix, suffix = C.split_half(text)
        assert prefix + suffix == text
        assert prefix == prefix.rstrip() and suffix[0].isspace()
        assert abs(len(re.findall(r"\S+", prefix)) - len(re.findall(r"\S+", suffix))) <= 1  # halves by token


def test_compare_ignores_punctuation_and_measures_the_verbatim_run():
    assert C.compare("To die—to sleep, No more", "to die: to sleep; no more")["exact_run"] == 6
    r = C.compare("one two three four five six", "one two three nine ten eleven")
    assert (r["exact_run"], r["run_ratio"], r["memorized"]) == (3, 0.5, True)
    assert not C.compare("one two three four five six", "seven eight nine")["memorized"]
    assert C.compare("Janet’s ducks", "Janet's ducks")["exact_run"] == 2  # curly vs straight apostrophe


def test_load_items(tmp_path):
    long = "word " * 20
    f = tmp_path / "bench.jsonl"
    f.write_text("\n".join([json.dumps({"id": "q1", "question": long}), json.dumps({"prompt": "too short"}),
                            "not json", json.dumps({"text": long + "x", "id": 7}), ""]))
    items, notes = C.load_items([str(f), str(tmp_path / "missing.jsonl")], limit=10)
    assert sorted(i for i, _ in items) == ["bench:7", "bench:q1"]
    assert any("not JSON" in n for n in notes) and any("missing" in n for n in notes)
    many = tmp_path / "many.jsonl"
    many.write_text("\n".join(json.dumps({"text": f"{long} {i}"}) for i in range(50)))
    a, _ = C.load_items([str(many)], limit=5)
    b, _ = C.load_items([str(many)], limit=5)
    assert len(a) == 5 and a == b  # a fixed sample, the same every run


@pytest.fixture
def run_suite(tmp_path, monkeypatch):
    def run(url, files=()):
        monkeypatch.chdir(tmp_path)
        for k in [k for k in os.environ if k.startswith("TEALEAF_")]:
            monkeypatch.delenv(k)
        monkeypatch.setenv("TEALEAF_BASE_URL", url)
        monkeypatch.setenv("TEALEAF_DB", str(tmp_path / "t.db"))
        cfg = load_config()
        cfg.run.contamination_files = list(files)
        store = Store(cfg.db_path)
        rid = asyncio.run(Runner(cfg, store).run(select_cases(["contamination"])))
        return store, rid, {r["case_id"]: r for r in store.get_results(rid)}
    return run


@pytest.mark.parametrize("flavor", ["llamacpp", "ollama"])
def test_suite_end_to_end(run_suite, flavor, llamacpp_url, mock_url, tmp_path):
    bench = tmp_path / "mine.jsonl"
    bench.write_text("\n".join([json.dumps({"id": "famous", "text": C.KNOWN[0][1]}),
                                json.dumps({"id": "fresh", "text": C.NOVEL[2][1]})]))
    store, rid, res = run_suite(llamacpp_url if flavor == "llamacpp" else mock_url, [str(bench)])
    known, novel, bm, mine = (res[f"contamination.{c}"] for c in ("known_texts", "novel_controls", "benchmarks",
                                                                     "your_items"))
    method = "raw" if flavor == "llamacpp" else "chat"
    assert known["metrics"]["method"] == method and known["metrics"]["memorized"] == len(C.KNOWN)
    assert novel["metrics"]["memorized"] == 0 and novel["score"] > 0.8 and not novel["flags"]
    assert bm["score"] == 0.0 and {f["name"] for f in bm["flags"]} == {"benchmark_memorized"}
    assert mine["score"] == 0.5 and [x["id"] for x in mine["metrics"]["items"] if x["memorized"]] == ["mine:famous"]
    if flavor == "llamacpp":  # the raw prompt is the bare text, with no chat template around it
        first = store.get_messages(rid, "contamination.known_texts")[0]
        assert first["meta"]["raw_prompt"] and C.KNOWN[0][1].startswith(first["content"])


def test_without_files_it_explains(run_suite, llamacpp_url):
    _, _, res = run_suite(llamacpp_url)
    r = res["contamination.your_items"]
    assert r["status"] == "observed" and "contamination_files" in r["notes"][0]


def test_cli_flag_adds_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _apply_overrides(load_config(), argparse.Namespace(contamination_file=["a.jsonl", "b.jsonl"]))
    assert cfg.run.contamination_files == ["a.jsonl", "b.jsonl"]
