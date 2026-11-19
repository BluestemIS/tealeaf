"""Configuration: harness.toml + environment overrides."""
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
    max_tokens: int = 2048
    timeout: float = 300.0
    stream: bool = True
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


@dataclass
class OpikSettings:
    enabled: bool = False
    url: str = "http://opik.home.lan/api"
    project: str = "llm-test-harness"
    workspace: str = "default"
    api_key: str = ""


@dataclass
class Config:
    target: Endpoint
    judge: Endpoint | None  # None -> the target judges itself
    run: RunSettings
    opik: OpikSettings
    db_path: str
    source: str | None = None

    def judge_endpoint(self) -> Endpoint:
        return self.judge or replace(self.target, temperature=0.0)

    def public(self) -> dict:
        return {
            "target": self.target.public(),
            "judge": self.judge.public() if self.judge else None,
            "run": asdict(self.run),
            "opik": {k: v for k, v in asdict(self.opik).items() if k != "api_key"},
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
        else:
            val = raw
        setattr(obj, f.name, val)


def find_config(explicit: str | None = None) -> Path | None:
    candidates = [explicit] if explicit else [
        os.environ.get("LLMH_CONFIG"),
        "harness.toml",
        str(Path.home() / ".config" / "llmharness" / "harness.toml"),
    ]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser().resolve()
    if explicit:
        raise FileNotFoundError(explicit)
    return None


def load_config(path: str | None = None) -> Config:
    """Load config. Precedence: env vars (LLMH_*, LLMH_JUDGE_*) > harness.toml > defaults."""
    cfg_path = find_config(path)
    data: dict = {}
    if cfg_path:
        data = tomllib.loads(cfg_path.read_text())

    target = _apply(Endpoint(), data.get("target", {}))
    _env(target, "LLMH_")

    judge = None
    jdata = dict(data.get("judge", {}))
    if any(os.environ.get(f"LLMH_JUDGE_{k}") for k in ("BASE_URL", "MODEL")) or jdata.get("base_url") or jdata.get("model"):
        judge = replace(target, temperature=0.0, extra={})
        _apply(judge, jdata)
        _env(judge, "LLMH_JUDGE_")

    run = _apply(RunSettings(), data.get("run", {}))
    opik = _apply(OpikSettings(), data.get("opik", {}))
    _env(opik, "LLMH_OPIK_")

    base = cfg_path.parent if cfg_path else Path.cwd()
    db = os.environ.get("LLMH_DB") or data.get("db_path") or "data/harness.db"
    db_path = str((base / db).resolve()) if not Path(db).is_absolute() else db

    return Config(target=target, judge=judge, run=run, opik=opik, db_path=db_path,
                  source=str(cfg_path) if cfg_path else None)
