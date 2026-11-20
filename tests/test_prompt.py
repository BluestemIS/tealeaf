"""The model's own prompt: chat template and built-in system prompt, from llama.cpp / Ollama."""
import asyncio

from conftest import serve_mock
from tealeaf.config import Endpoint
from tealeaf.discovery import _has_text, model_prompt
from tealeaf.mock_server import MOCK_SYSTEM, MOCK_TEMPLATE


def _prompt(url, model):
    return asyncio.run(model_prompt(Endpoint(base_url=url), model))


def test_ollama_system_and_template(mock_url):
    p = _prompt(mock_url, "mock-small:1b")
    assert (p.server, p.available, p.system, p.template) == ("ollama", True, MOCK_SYSTEM, MOCK_TEMPLATE)
    assert _prompt(mock_url, "mock-thinker:8b").system == ""  # no SYSTEM in its Modelfile


def test_llamacpp_renders_template():
    with serve_mock("llamacpp") as url:
        p = _prompt(url, "mock-small:1b")
    assert (p.server, p.available, p.template) == ("llamacpp", True, MOCK_TEMPLATE)
    assert MOCK_SYSTEM in p.system and p.preamble == p.system
    assert "{your message}" in p.rendered


def test_unsupported_server_is_not_available():
    with serve_mock("openai") as url:
        p = _prompt(url, "mock-small:1b")
    assert not p.available and p.system is None and "doesn't expose" in p.note


def test_unreachable_server_never_raises():
    p = _prompt("http://127.0.0.1:9/v1", "x")
    assert not p.available and p.note


def test_preamble_text_detection():
    assert not _has_text("<|im_start|>user\n")
    assert not _has_text("<bos><start_of_turn>user\n")
    assert not _has_text("<s>[INST] ")
    assert not _has_text("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n")
    assert _has_text("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                     "Cutting Knowledge Date: December 2023\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")
