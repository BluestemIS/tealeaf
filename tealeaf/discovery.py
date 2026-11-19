"""Discover the models an endpoint serves, with whatever metadata the server exposes.

Every OpenAI-compatible server answers GET /v1/models with ids only. When the server is Ollama or LM Studio we
also ask their native APIs for size, quantization, context length, capabilities and whether the model is loaded.
Enrichment is best-effort: any failure just leaves those fields empty.
"""
from __future__ import annotations

import asyncio
import fnmatch
import re
from dataclasses import asdict, dataclass, field

import httpx

from .config import Endpoint

EMBED_HINT = re.compile(r"embed|bge-|e5-|nomic-embed|minilm|rerank", re.I)


@dataclass
class ModelInfo:
    id: str
    kind: str = "llm"  # llm | vlm | embedding
    server: str = "openai"  # openai | ollama | lmstudio | llamacpp | vllm
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    context_length: int | None = None
    loaded: bool | None = None
    capabilities: list[str] = field(default_factory=list)
    owned_by: str | None = None

    @property
    def chat_capable(self) -> bool:
        return self.kind != "embedding"

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        parts = [self.family, self.parameter_size, self.quantization]
        if self.size_bytes:
            parts.append(f"{self.size_bytes / 1e9:.1f} GB")
        if self.context_length:
            parts.append(f"ctx {self.context_length:,}")
        if "thinking" in self.capabilities:
            parts.append("thinking")
        if self.loaded:
            parts.append("loaded")
        if self.kind != "llm":
            parts.append(self.kind)
        return " · ".join(p for p in parts if p)


class DiscoveryError(RuntimeError):
    pass


def _root(base_url: str) -> str:
    """http://host:11434/v1 -> http://host:11434"""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


async def discover(ep: Endpoint, *, enrich: bool = True, timeout: float = 8.0) -> list[ModelInfo]:
    headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as http:
        try:
            r = await http.get(ep.base_url.rstrip("/") + "/models")
            r.raise_for_status()
            data = r.json().get("data", [])
        except (httpx.HTTPError, ValueError) as e:
            raise DiscoveryError(f"cannot list models at {ep.base_url}: {e}") from e
        models = {m["id"]: ModelInfo(id=m["id"], owned_by=m.get("owned_by")) for m in data if m.get("id")}
        for m in models.values():
            if EMBED_HINT.search(m.id):
                m.kind = "embedding"
        _enrich_inline(data, models)
        if enrich and models:
            root = _root(ep.base_url)
            if not await _enrich_lmstudio(http, root, models):
                await _enrich_ollama(http, root, models)
    return sorted(models.values(), key=lambda m: (m.kind == "embedding", not m.loaded, m.id.lower()))


def _human_params(n: int) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def _enrich_inline(rows: list[dict], models: dict[str, ModelInfo]) -> None:
    """Details some servers put straight into /v1/models: llama.cpp ("meta"), vLLM ("max_model_len")."""
    for row in rows:
        m = models.get(row.get("id"))
        if not m:
            continue
        meta = row.get("meta") or {}
        if meta or row.get("owned_by") == "llamacpp":
            m.server = "llamacpp"
            m.loaded = True  # llama.cpp only lists what it is serving
            m.context_length = meta.get("n_ctx") or m.context_length
            if isinstance(meta.get("n_params"), int):
                m.parameter_size = _human_params(meta["n_params"])
            if isinstance(meta.get("size"), int):
                m.size_bytes = meta["size"]
            if meta.get("ftype"):
                m.quantization = str(meta["ftype"]).split(" - ")[0]
        if isinstance(row.get("max_model_len"), int):  # vLLM
            m.server = "vllm"
            m.context_length = row["max_model_len"]
        mods = (row.get("architecture") or {}).get("input_modalities") or []
        if "image" in mods and m.kind == "llm":
            m.kind = "vlm"


async def _enrich_lmstudio(http: httpx.AsyncClient, root: str, models: dict[str, ModelInfo]) -> bool:
    try:
        r = await http.get(f"{root}/api/v0/models")
        if r.status_code != 200:
            return False
        rows = r.json().get("data", [])
    except (httpx.HTTPError, ValueError):
        return False
    if not rows:
        return False
    for row in rows:
        m = models.get(row.get("id"))
        if not m:
            continue
        m.server = "lmstudio"
        t = row.get("type")
        m.kind = {"embeddings": "embedding", "vlm": "vlm"}.get(t, "llm" if t == "llm" else m.kind)
        m.family = row.get("arch") or m.family
        m.quantization = row.get("quantization")
        m.context_length = row.get("max_context_length")
        m.loaded = row.get("state") == "loaded"
        m.owned_by = row.get("publisher") or m.owned_by
    return True


async def _enrich_ollama(http: httpx.AsyncClient, root: str, models: dict[str, ModelInfo]) -> bool:
    try:
        r = await http.get(f"{root}/api/tags")
        if r.status_code != 200:
            return False
        tags = r.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return False
    loaded: set[str] = set()
    try:
        ps = await http.get(f"{root}/api/ps")
        if ps.status_code == 200:
            loaded = {x.get("name") or x.get("model") for x in ps.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        pass
    for t in tags:
        name = t.get("name") or t.get("model")
        if not name:
            continue
        m = models.get(name) or models.get(name.removesuffix(":latest"))
        if not m:
            continue
        d = t.get("details") or {}
        if m.server == "llamacpp":  # llama.cpp also answers /api/tags, but with empty fields
            continue
        m.server = "ollama"
        m.family = d.get("family")
        m.parameter_size = d.get("parameter_size")
        m.quantization = d.get("quantization_level")
        m.size_bytes = t.get("size")
        m.loaded = name in loaded

    async def show(m: ModelInfo):
        try:
            r = await http.post(f"{root}/api/show", json={"model": m.id})
            if r.status_code != 200:
                return
            j = r.json()
        except (httpx.HTTPError, ValueError):
            return
        m.capabilities = list(j.get("capabilities") or [])
        if "embedding" in m.capabilities and "completion" not in m.capabilities:
            m.kind = "embedding"
        elif "vision" in m.capabilities:
            m.kind = "vlm"
        for k, v in (j.get("model_info") or {}).items():
            if k.endswith(".context_length") and isinstance(v, int):
                m.context_length = v

    ollama_models = [m for m in models.values() if m.server == "ollama"]
    await asyncio.gather(*(show(m) for m in ollama_models))
    return bool(ollama_models)


def match_models(spec: str | list[str] | None, available: list[ModelInfo]) -> list[str]:
    """Resolve a model spec against discovered models.

    'all' -> every chat-capable model; globs ('qwen*') and comma lists are matched; an exact id that the server
    did not list is kept as-is (some servers load models on demand).
    """
    if not spec:
        return []
    parts = spec if isinstance(spec, list) else [p.strip() for p in spec.split(",")]
    chat = [m.id for m in available if m.chat_capable]
    out: list[str] = []
    for p in filter(None, parts):
        if p.lower() == "all":
            hits = chat
        elif any(c in p for c in "*?["):
            hits = [i for i in chat if fnmatch.fnmatch(i, p)]
        else:
            hits = [p]
        out += [h for h in hits if h not in out]
    return out
