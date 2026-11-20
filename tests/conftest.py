"""Shared fixtures."""
import socket
import threading
import time

import pytest
import uvicorn

from tealeaf.mock_server import create_mock_app


@pytest.fixture(scope="module")
def mock_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_mock_app(0), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
