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


MOCK_MODELS = [
    # id, family, params, quant, bytes, ctx, capabilities, loaded
    ("mock-small:1b", "llama", "1.2B", "Q4_K_M", 808_000_000, 131072, ["completion", "tools"], True),
    ("mock-thinker:8b", "qwen3", "8.2B", "Q4_K_M", 5_200_000_000, 40960, ["completion", "tools", "thinking"], False),
    ("mock-vision:4b", "gemma3", "4.3B", "Q4_K_M", 3_300_000_000, 131072, ["completion", "vision"], False),
    ("nomic-embed-text:latest", "nomic-bert", "137M", "F16", 274_000_000, 2048, ["embedding"], False),
]


MOCK_SYSTEM = "You are MockBot, a helpful assistant made by the tealeaf authors."
MOCK_TEMPLATE = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"


def create_mock_app(delay: float = 0.01, flavor: str = "ollama") -> FastAPI:
    """flavor: 'ollama', 'lmstudio' or 'llamacpp' adds that server's native API; 'openai' serves /v1/models only."""
    app = FastAPI(title="mock llm")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m[0], "object": "model", "owned_by": "mock"} for m in MOCK_MODELS]}

    if flavor == "ollama":
        @app.get("/api/tags")
        def tags():
            return {"models": [{"name": m[0], "model": m[0], "size": m[4], "details": {
                "family": m[1], "parameter_size": m[2], "quantization_level": m[3]}} for m in MOCK_MODELS]}

        @app.get("/api/ps")
        def ps():
            return {"models": [{"name": m[0]} for m in MOCK_MODELS if m[7]]}

        @app.post("/api/show")
        async def show(req: Request):
            name = (await req.json()).get("model")
            m = next((x for x in MOCK_MODELS if x[0] == name), None)
            if not m:
                return JSONResponse({"error": "not found"}, status_code=404)
            return {"capabilities": m[6], "model_info": {f"{m[1]}.context_length": m[5]},
                    "template": MOCK_TEMPLATE, "system": MOCK_SYSTEM if m[0] == "mock-small:1b" else ""}

    if flavor == "llamacpp":
        @app.get("/props")
        def props():
            return {"chat_template": MOCK_TEMPLATE, "model_path": "/models/mock.gguf"}

        @app.post("/apply-template")
        async def apply_template(req: Request):
            msgs = (await req.json()).get("messages", [])
            out = "" if msgs and msgs[0].get("role") == "system" else f"<|im_start|>system\n{MOCK_SYSTEM}<|im_end|>\n"
            out += "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
            return {"prompt": out + "<|im_start|>assistant\n"}

    if flavor == "lmstudio":
        @app.get("/api/v0/models")
        def lms():
            return {"data": [{"id": m[0], "type": "embeddings" if "embedding" in m[6] else
                              "vlm" if "vision" in m[6] else "llm", "arch": m[1], "quantization": m[3],
                              "state": "loaded" if m[7] else "not-loaded", "max_context_length": m[5],
                              "publisher": "mock"} for m in MOCK_MODELS]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        model = body.get("model") or MOCK_MODELS[0][0]
        if model not in {m[0] for m in MOCK_MODELS}:
            return JSONResponse({"error": {"message": f"model '{model}' not found"}}, status_code=404)
        text = _answer(body.get("messages", []))
        ptoks = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 4
        usage = {"prompt_tokens": ptoks, "completion_tokens": max(1, len(text) // 4)}
        if not body.get("stream"):
            return JSONResponse({"id": "x", "model": model, "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage})

        async def gen():
            for piece in re.findall(r"\S+\s*", text) or [text]:
                await asyncio.sleep(delay)
                yield "data: " + json.dumps({"id": "x", "model": model, "created": int(time.time()), "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"id": "x", "model": model, "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
