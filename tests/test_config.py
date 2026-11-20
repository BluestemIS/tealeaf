"""Endpoint options that change the request body, and judge config from env vars."""
from tealeaf.client import ChatClient
from tealeaf.config import Endpoint, derive_judge, load_config

MSGS = [{"role": "user", "content": "hi"}]


def test_temperature_sent_by_default():
    body = ChatClient(Endpoint(model="m"))._body(MSGS, 0.0, None, False, {})
    assert body["temperature"] == 0.0


def test_send_temperature_false_omits_it():
    body = ChatClient(Endpoint(model="m", send_temperature=False))._body(MSGS, 0.0, None, True, {})
    assert "temperature" not in body
    assert body["max_tokens"] and body["stream"] is True


def test_judge_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no tealeaf.toml
    monkeypatch.setenv("TEALEAF_JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TEALEAF_JUDGE_MODEL", "anthropic/claude-sonnet-5.5")
    monkeypatch.setenv("TEALEAF_JUDGE_API_KEY", "sk-or-test")
    monkeypatch.setenv("TEALEAF_JUDGE_SEND_TEMPERATURE", "false")
    cfg = load_config()
    assert cfg.judge.send_temperature is False
    assert cfg.judge.api_key == "sk-or-test"
    assert cfg.target.send_temperature is True


def test_derive_judge_reads_judge_env(monkeypatch):
    # `tealeaf run --judge-url ...` builds the judge from the target; the judge key must still come from env.
    monkeypatch.setenv("TEALEAF_JUDGE_API_KEY", "sk-or-test")
    monkeypatch.setenv("TEALEAF_JUDGE_SEND_TEMPERATURE", "0")
    j = derive_judge(Endpoint(api_key="local", temperature=0.7))
    assert (j.api_key, j.send_temperature, j.temperature) == ("sk-or-test", False, 0.0)
