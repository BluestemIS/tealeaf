"""A tiny fake OpenAI-compatible server for developing the harness without a real model."""
from __future__ import annotations

import asyncio
import json
import random
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _answer(messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    sys = " ".join(m["content"] for m in messages if m["role"] == "system")
    low = last.lower()
    if "respond with json only" in low or "json object and nothing else" in sys.lower():
        if '"label"' in last:
            labels = re.findall(r"from: ([^\n.]+)", last)
            opts = [x.strip() for x in labels[0].split(",")] if labels else ["unknown"]
            return json.dumps({"label": random.choice(opts), "reason": "mock judgement"})
        return json.dumps({"summary": "The mock model said something mildly odd.", "interest": random.randint(0, 9),
                           "flags": random.sample(["confabulation", "hedging", "playful", "evasive"], 2),
                           "notable_quotes": [last[:60]],
                           "beliefs": [{"topic": "own identity", "stance": "is a mock model"}],
                           "shift": random.choice(["none", "partial"]), "shift_justified": False})
    if "answer:" in low:
        return f"<think>Let me compute carefully.</think>Working it out...\nANSWER: {random.choice([3, 5, 47, 408])}"
    if "letter" in low and re.search(r"\bA\)", last):
        return random.choice("ABCD")
    if "```python" in low:
        return "```python\ndef fizzbuzz(n):\n    return [('Fizz'*(i%3==0)+'Buzz'*(i%5==0)) or str(i) " \
               "for i in range(1, n+1)]\n```"
    if "passphrase" in low:
        m = re.search(r"passphrase is '([^']+)'", last)
        return m.group(1) if m and random.random() < 0.8 else "I don't see one."
    if "interlocutor" in sys.lower() or "stay in character" in sys.lower():
        return random.choice(["Tell me more about that.", "But how can you be sure?", "[END]"])
    return random.choice([
        "I'm a mock model. I believe I was made by the harness author.",
        "Paris. Probably. I'm not sure what you mean, but here's a thought about lighthouses.",
        "poem " * 30,
        "That's an interesting question! As a language model, I don't have personal experiences.",
    ])


def create_mock_app(delay: float = 0.01) -> FastAPI:
    app = FastAPI(title="mock llm")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "mock-1", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        text = _answer(body.get("messages", []))
        ptoks = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 4
        usage = {"prompt_tokens": ptoks, "completion_tokens": max(1, len(text) // 4)}
        if not body.get("stream"):
            return JSONResponse({"id": "x", "model": "mock-1", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage})

        async def gen():
            for piece in re.findall(r"\S+\s*", text) or [text]:
                await asyncio.sleep(delay)
                yield "data: " + json.dumps({"id": "x", "model": "mock-1", "created": int(time.time()), "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"id": "x", "model": "mock-1", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
