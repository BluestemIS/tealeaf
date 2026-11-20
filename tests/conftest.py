"""Shared fixtures."""
import contextlib
import socket
import threading
import time

import pytest
import uvicorn

from tealeaf.mock_server import create_mock_app


@contextlib.contextmanager
def serve_mock(flavor: str = "ollama"):
    """Run the mock model server in a thread; yields its /v1 base URL."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_mock_app(0, flavor), host="127.0.0.1", port=port,
                                           log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True


@pytest.fixture(scope="module")
def mock_url():
    with serve_mock() as url:
        yield url
