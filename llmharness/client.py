"""Async client for OpenAI-compatible /chat/completions endpoints."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

from .config import Endpoint
from .scoring import strip_think


class ClientError(RuntimeError):
    pass


@dataclass
class Reply:
    content: str  # visible answer, <think> blocks removed
    reasoning: str = ""  # reasoning_content / <think> text, if any
    raw_content: str = ""
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_estimated: bool = False
    latency: float = 0.0
    ttft: float | None = None
    finish_reason: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_sec(self) -> float | None:
        if not self.completion_tokens:
            return None
        gen = self.latency - (self.ttft or 0.0)
        return self.completion_tokens / gen if gen > 0 else None

    def stats(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_estimated": self.tokens_estimated,
            "latency": round(self.latency, 3),
            "ttft": round(self.ttft, 3) if self.ttft is not None else None,
            "tokens_per_sec": round(self.tokens_per_sec, 2) if self.tokens_per_sec else None,
            "finish_reason": self.finish_reason,
        }


DeltaCallback = Callable[[str, str], Awaitable[None] | None]


class ChatClient:
    def __init__(self, ep: Endpoint, name: str = "target"):
        self.ep = ep
        self.name = name
        headers = {"Content-Type": "application/json"}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        self._http = httpx.AsyncClient(
            base_url=ep.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(ep.timeout, connect=10.0),
        )
        self._stream_options_ok = True

    @property
    def model(self) -> str:
        return self.ep.model

    async def aclose(self):
        await self._http.aclose()

    async def list_models(self) -> list[str]:
        try:
            r = await self._http.get("/models")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ClientError(f"{self.name}: cannot list models at {self.ep.base_url}: {e}") from e
        return [m.get("id") for m in r.json().get("data", []) if m.get("id")]

    async def resolve_model(self) -> str:
        if not self.ep.model:
            models = await self.list_models()
            if not models:
                raise ClientError(f"{self.name}: no models reported by {self.ep.base_url}/models")
            self.ep.model = models[0]
        return self.ep.model

    def _body(self, messages, temperature, max_tokens, stream, extra) -> dict:
        body = {
            "model": self.ep.model,
            "messages": messages,
            "temperature": self.ep.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.ep.max_tokens,
            "stream": stream,
        }
        if stream and self._stream_options_ok:
            body["stream_options"] = {"include_usage": True}
        body.update(self.ep.extra)
        body.update(extra)
        return body

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
        **extra,
    ) -> Reply:
        await self.resolve_model()
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                if self.ep.stream:
                    return await self._chat_stream(messages, temperature, max_tokens, on_delta, extra)
                return await self._chat_once(messages, temperature, max_tokens, extra)
            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response is not None else ""
                if e.response is not None and e.response.status_code == 400 and self._stream_options_ok and self.ep.stream:
                    self._stream_options_ok = False  # some servers reject stream_options; retry without
                    last_err = e
                    continue
                raise ClientError(f"{self.name}: HTTP {e.response.status_code}: {body}") from e
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                last_err = e
                await asyncio.sleep(1.5 * (attempt + 1))
            except httpx.TimeoutException as e:
                raise ClientError(f"{self.name}: request timed out after {self.ep.timeout}s") from e
        raise ClientError(f"{self.name}: request failed: {last_err!r}")

    async def _chat_once(self, messages, temperature, max_tokens, extra) -> Reply:
        t0 = time.perf_counter()
        r = await self._http.post("/chat/completions", json=self._body(messages, temperature, max_tokens, False, extra))
        r.raise_for_status()
        data = r.json()
        latency = time.perf_counter() - t0
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        raw = msg.get("content") or ""
        content, think = strip_think(raw)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or think
        usage = data.get("usage") or {}
        rep = Reply(
            content=content, reasoning=reasoning, raw_content=raw, model=data.get("model", self.ep.model),
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            latency=latency, finish_reason=choice.get("finish_reason"),
        )
        _estimate(rep)
        return rep

    async def _chat_stream(self, messages, temperature, max_tokens, on_delta, extra) -> Reply:
        t0 = time.perf_counter()
        ttft = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        finish = None
        model = self.ep.model
        body = self._body(messages, temperature, max_tokens, True, extra)
        async with self._http.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code >= 400:
                await r.aread()
                r.raise_for_status()
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ClientError(f"{self.name}: server error: {chunk['error']}")
                model = chunk.get("model") or model
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    c = delta.get("content") or ""
                    rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (c or rc) and ttft is None:
                        ttft = time.perf_counter() - t0
                    if c:
                        content_parts.append(c)
                    if rc:
                        reasoning_parts.append(rc)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if on_delta and (c or rc):
                        res = on_delta("".join(content_parts), "".join(reasoning_parts))
                        if asyncio.iscoroutine(res):
                            await res
        latency = time.perf_counter() - t0
        raw = "".join(content_parts)
        content, think = strip_think(raw)
        reasoning = "".join(reasoning_parts) or think
        rep = Reply(
            content=content, reasoning=reasoning, raw_content=raw, model=model,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            latency=latency, ttft=ttft, finish_reason=finish,
        )
        _estimate(rep)
        return rep


def _estimate(rep: Reply):
    """Fill in a rough token count (~4 chars/token) when the server reports no usage."""
    if rep.completion_tokens is None:
        text = rep.raw_content + (rep.reasoning if rep.reasoning not in rep.raw_content else "")
        rep.completion_tokens = max(1, len(text) // 4) if text else 0
        rep.tokens_estimated = True
