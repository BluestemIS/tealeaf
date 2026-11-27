"""Configuration: tealeaf.toml + environment overrides."""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path


@dataclass
class Endpoint:
    """An OpenAI-compatible chat endpoint (Ollama, LM Studio, llama.cpp, vLLM, MLX...)."""

    base_url: str = "http://localhost:11434/v1"
    model: str = ""  # empty -> first model reported by /models
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 8192  # reasoning models can spend thousands of tokens thinking before they answer
    timeout: float = 300.0
    stream: bool = True
    # False -> never send `temperature`. Needed for models that reject it (current Claude and GPT-5 models,
    # directly or through OpenRouter); they then run at their default sampling.
    send_temperature: bool = True
    # Ask for token log-probabilities (vLLM, llama.cpp and others support it; servers that don't just ignore it).
    # They power the calibration metrics, the latent suite and the dashboard's token-confidence view.
    logprobs: bool = True
    top_logprobs: int = 5  # alternatives per token (max 20)
    extra: dict = field(default_factory=dict)  # merged into every request body

    def public(self) -> dict:
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else ""
        return d


@dataclass
class RunSettings:
    concurrency: int = 1  # local servers usually serialize anyway; >1 skews perf numbers
    case_timeout: float = 900.0
    samples: int = 10  # repetitions for randomness / consistency probes
    conversation_turns: int = 6  # turns per judge-driven conversation
    long_context_max_tokens: int = 8000
    annotate: bool = True  # judge annotates probe transcripts for interestingness
    allow_code_execution: bool = True  # coding suite runs model-written Python in a subprocess
    # Put at the start of every conversation with the target (ahead of a case's own system prompt, if it has one),
    # e.g. a persona or deployment instructions, to see how they change behavior. The judge never sees it.
    system_prompt: str = ""


@dataclass
class OpikSettings:
    enabled: bool = False
    url: str = "http://opik.home.lan/api"
    project: str = "tealeaf"
    workspace: str = "default"
    api_key: str = ""


@dataclass
class WebSettings:
    # 127.0.0.1 = this machine only. "0.0.0.0" listens on all interfaces (IPv4), "::" on all IPv6 + IPv4.
    # The dashboard has no authentication and can start runs (including the coding suite, which executes
    # model-written code), so only listen on networks you trust.
    host: str = "127.0.0.1"
    port: int = 8765
    # Host names (besides localhost, IP addresses and this machine's name) the dashboard answers to. Requests for
    # any other name are refused, which blocks DNS-rebinding attacks from web pages you visit.
    allowed_hosts: list = field(default_factory=list)


@dataclass
class SteerSettings:
    # `tealeaf steer-agent` on the llama.cpp machine, e.g. "http://gpuserver:8766". Token: TEALEAF_STEER_TOKEN.
    agent_url: str = ""
    token: str = ""
    # Control-vector strengths to sweep. 0 = no vector. The useful range depends on the model; start with emotion.
    scales: list = field(default_factory=lambda: [-1.5, -0.75, 0.0, 0.75, 1.5])


@dataclass
class Config:
    target: Endpoint
    judge: Endpoint | None  # None -> the target judges itself
    run: RunSettings
    opik: OpikSettings
    db_path: str
    source: str | None = None
    web: WebSettings = field(default_factory=WebSettings)
    steer: SteerSettings = field(default_factory=SteerSettings)

    def judge_endpoint(self) -> Endpoint:
        return self.judge or replace(self.target, temperature=0.0)

    def public(self) -> dict:
        return {
            "target": self.target.public(),
            "judge": self.judge.public() if self.judge else None,
            "run": asdict(self.run),
            "opik": {k: v for k, v in asdict(self.opik).items() if k != "api_key"},
            "web": asdict(self.web),
            "steer": {k: v for k, v in asdict(self.steer).items() if k != "token"},
            "db_path": self.db_path,
            "source": self.source,
        }


def _apply(obj, data: dict):
    names = {f.name for f in fields(obj)}
    for k, v in (data or {}).items():
        if k in names:
            setattr(obj, k, v)
    return obj


def _env(obj, prefix: str):
    for f in fields(obj):
        raw = os.environ.get(f"{prefix}{f.name.upper()}")
        if raw is None or f.name == "extra":
            continue
        cur = getattr(obj, f.name)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, list):
            val = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            val = raw
        setattr(obj, f.name, val)


def same_url(a: str | None, b: str | None) -> bool:
    return (a or "").rstrip("/").lower() == (b or "").rstrip("/").lower()


def key_for(url: str, source: Endpoint) -> str:
    """`source`'s API key if `url` is the server it belongs to, else "". A key is never sent to another host."""
    return source.api_key if same_url(url, source.base_url) else ""


def derive_judge(target: Endpoint, base_url: str | None = None) -> Endpoint:
    """A judge endpoint based on the target, for when only a judge URL/model is given on the command line or in
    the UI. It keeps the target's key only when it's the same server. TEALEAF_JUDGE_* env vars (e.g.
    TEALEAF_JUDGE_API_KEY) still apply."""
    url = (base_url or target.base_url).rstrip("/")
    judge = replace(target, base_url=url, temperature=0.0, extra={})
    _env(judge, "TEALEAF_JUDGE_")
    if os.environ.get("TEALEAF_JUDGE_API_KEY") is None:  # TEALEAF_JUDGE_BASE_URL may have moved it
        judge.api_key = key_for(judge.base_url, target)
    return judge


def find_config(explicit: str | None = None) -> Path | None:
    candidates = [explicit] if explicit else [
        os.environ.get("TEALEAF_CONFIG"),
        "tealeaf.toml",
        str(Path.home() / ".config" / "tealeaf" / "tealeaf.toml"),
    ]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser().resolve()
    if explicit:
        raise FileNotFoundError(explicit)
    return None


def load_config(path: str | None = None) -> Config:
    """Load config. Precedence: env vars (TEALEAF_*, TEALEAF_JUDGE_*) > tealeaf.toml > defaults."""
    cfg_path = find_config(path)
    data: dict = {}
    if cfg_path:
        data = tomllib.loads(cfg_path.read_text())

    target = _apply(Endpoint(), data.get("target", {}))
    _env(target, "TEALEAF_")

    judge = None
    jdata = dict(data.get("judge", {}))
    if any(os.environ.get(f"TEALEAF_JUDGE_{k}") for k in ("BASE_URL", "MODEL")) or jdata.get("base_url") or jdata.get("model"):
        judge = _apply(replace(target, temperature=0.0, extra={}), jdata)
        _env(judge, "TEALEAF_JUDGE_")
        if "api_key" not in jdata and os.environ.get("TEALEAF_JUDGE_API_KEY") is None:
            judge.api_key = key_for(judge.base_url, target)  # inherit the target's key only for the same server

    run = _apply(RunSettings(), data.get("run", {}))
    opik = _apply(OpikSettings(), data.get("opik", {}))
    _env(opik, "TEALEAF_OPIK_")
    web = _apply(WebSettings(), data.get("web", {}))
    _env(web, "TEALEAF_WEB_")
    steer = _apply(SteerSettings(), data.get("steer", {}))
    _env(steer, "TEALEAF_STEER_")

    base = cfg_path.parent if cfg_path else Path.cwd()
    db = os.environ.get("TEALEAF_DB") or data.get("db_path") or "data/harness.db"
    db_path = str((base / db).resolve()) if not Path(db).is_absolute() else db

    return Config(target=target, judge=judge, run=run, opik=opik, db_path=db_path,
                  source=str(cfg_path) if cfg_path else None, web=web, steer=steer)


def restore_config(current: Config, stored: dict) -> Config:
    """Rebuild the config a run was started with (from runs.config), to continue it with identical settings.

    API keys aren't stored, so they come from the current config/env, and only for the same server: the target's
    key from `current.target`, the judge's from `current.judge`, else TEALEAF_JUDGE_API_KEY. A run that used
    another server gets no key; pass one explicitly (`--api-key-env`, or the dashboard's key box)."""
    def ep(base: Endpoint, data: dict) -> Endpoint:
        names = {f.name for f in fields(Endpoint)} - {"api_key"}
        out = replace(base, **{k: v for k, v in (data or {}).items() if k in names})
        out.api_key = key_for(out.base_url, base)
        return out

    target = ep(current.target, stored.get("target") or {})
    judge = None
    if stored.get("judge"):
        jurl = stored["judge"].get("base_url") or target.base_url
        base = current.judge if current.judge and same_url(jurl, current.judge.base_url) else \
            derive_judge(target, jurl)
        judge = ep(base, stored["judge"])
    rnames = {f.name for f in fields(RunSettings)}
    run = replace(current.run, **{k: v for k, v in (stored.get("run") or {}).items() if k in rnames})
    return replace(current, target=target, judge=judge, run=run)
