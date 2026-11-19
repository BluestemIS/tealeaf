"""Model discovery against mock Ollama, LM Studio and plain OpenAI-compatible servers."""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from tealeaf.config import Endpoint
from tealeaf.discovery import ModelInfo, discover, match_models
from tealeaf.mock_server import create_mock_app


def _serve(flavor):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_mock_app(0, flavor), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    return server, f"http://127.0.0.1:{port}/v1"


@pytest.fixture(scope="module", params=["ollama", "lmstudio", "openai"])
def endpoint(request):
    server, url = _serve(request.param)
    yield request.param, url
    server.should_exit = True


def test_discover(endpoint):
    flavor, url = endpoint
    models = {m.id: m for m in asyncio.run(discover(Endpoint(base_url=url)))}
    assert set(models) == {"mock-small:1b", "mock-thinker:8b", "mock-vision:4b", "nomic-embed-text:latest"}
    assert models["nomic-embed-text:latest"].kind == "embedding"
    assert not models["nomic-embed-text:latest"].chat_capable
    if flavor == "openai":
        assert models["mock-small:1b"].server == "openai"
        return
    small = models["mock-small:1b"]
    assert small.server == flavor
    assert small.loaded is True and models["mock-thinker:8b"].loaded is False
    assert small.context_length == 131072
    assert small.quantization == "Q4_K_M"
    assert models["mock-vision:4b"].kind == "vlm"
    if flavor == "ollama":
        assert small.parameter_size == "1.2B" and small.size_bytes == 808_000_000
        assert "thinking" in models["mock-thinker:8b"].capabilities
    # loaded chat models sort first, embeddings last
    order = list(models)
    assert order[0] == "mock-small:1b" and order[-1] == "nomic-embed-text:latest"


def test_match_models():
    avail = [ModelInfo("qwen3:8b"), ModelInfo("qwen3:32b"), ModelInfo("llama3.2:3b"),
             ModelInfo("nomic-embed-text", kind="embedding")]
    assert match_models("all", avail) == ["qwen3:8b", "qwen3:32b", "llama3.2:3b"]
    assert match_models("qwen3:*", avail) == ["qwen3:8b", "qwen3:32b"]
    assert match_models("llama3.2:3b, qwen3:8b", avail) == ["llama3.2:3b", "qwen3:8b"]
    assert match_models("not-listed", avail) == ["not-listed"]
    assert match_models("*embed*", avail) == []
    assert match_models("", avail) == []
