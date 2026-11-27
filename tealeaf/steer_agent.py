"""`tealeaf steer-agent`: runs on the llama.cpp machine and does what control vectors need locally.

llama.cpp builds a control vector with `llama-cvector-generator` (it reads the model's GGUF) and applies it only at
llama-server startup (`--control-vector-scaled FILE SCALE`). So this agent owns a second llama-server and restarts
it for each steering strength, and builds vectors from prompt pairs sent by `tealeaf steer`.

Everything needs the token. The agent only runs the two llama.cpp binaries, only loads GGUF files from its models
folder, and never uses a shell.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SERVER_FLAGS = ("--control-vector-scaled", "--control-vector-layer-range")
GENERATOR_FLAGS = ("--positive-file", "--negative-file", "--method")


class AgentFailure(RuntimeError):
    pass


def _find(name: str, bin_dir: str | None) -> str:
    path = shutil.which(name, path=bin_dir) if bin_dir else shutil.which(name)
    if not path:
        raise AgentFailure(f"{name} not found{f' in {bin_dir}' if bin_dir else ' on PATH'} (use --llama-bin)")
    return path


def _help(binary: str) -> str:
    try:
        p = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise AgentFailure(f"cannot run {binary} --help: {e}") from e
    return p.stdout + p.stderr


class Agent:
    def __init__(self, *, bin_dir: str | None, models_dir: str, default_model: str | None, data_dir: str,
                 server_host: str = "0.0.0.0", server_port: int = 8081, server_args: list[str] | None = None,
                 gen_args: list[str] | None = None, health_timeout: float = 600.0):
        self.server_bin = _find("llama-server", bin_dir)
        self.gen_bin = _find("llama-cvector-generator", bin_dir)
        # Check the flags this llama.cpp build actually has, rather than failing halfway through a sweep.
        for binary, flags in ((self.server_bin, SERVER_FLAGS), (self.gen_bin, GENERATOR_FLAGS)):
            text = _help(binary)
            missing = [f for f in flags if f not in text]
            if missing:
                raise AgentFailure(f"{binary} lacks {', '.join(missing)}; is this llama.cpp build too old?")
        self.models_dir = Path(models_dir).expanduser().resolve()
        if not self.models_dir.is_dir():
            raise AgentFailure(f"models folder not found: {self.models_dir}")
        self.default_model = default_model
        if default_model:
            self.model_path(default_model)  # fail now if it isn't there
        self.data_dir = Path(data_dir).expanduser().resolve()
        (self.data_dir / "vectors").mkdir(parents=True, exist_ok=True)
        self.server_host, self.server_port = server_host, server_port
        self.server_args, self.gen_args = list(server_args or []), list(gen_args or [])
        self.health_timeout = health_timeout
        self.proc: subprocess.Popen | None = None
        self.state: dict = {}
        self.lock = threading.Lock()  # one server start/stop or vector build at a time
        self.jobs: dict[str, dict] = {}

    # ---- paths ----------------------------------------------------------------
    def model_path(self, name: str | None) -> Path:
        name = name or self.default_model
        if not name:
            raise AgentFailure("no model given and the agent has no --model default")
        p = (self.models_dir / name).resolve()
        if self.models_dir not in p.parents:
            raise AgentFailure("model must be inside the agent's models folder")
        if p.suffix != ".gguf" or not p.is_file():
            raise AgentFailure(f"no such GGUF in the models folder: {name}")
        return p

    def vector_path(self, model: Path, name: str) -> Path:
        if not NAME_RE.match(name):
            raise AgentFailure("vector names may use letters, digits, '_', '-' and '.'")
        return self.data_dir / "vectors" / model.stem / f"{name}.gguf"

    def models(self) -> list[str]:
        return sorted(str(p.relative_to(self.models_dir)) for p in self.models_dir.rglob("*.gguf"))

    def vectors(self) -> list[dict]:
        out = []
        for p in sorted((self.data_dir / "vectors").glob("*/*.gguf")):
            meta = p.with_suffix(".json")
            info = json.loads(meta.read_text()) if meta.is_file() else {}
            out.append({"model_stem": p.parent.name, "name": p.stem, "created": p.stat().st_mtime, **info})
        return out

    # ---- the steered llama-server ----------------------------------------------
    def _log(self, name: str) -> Path:
        return self.data_dir / f"{name}.log"

    def _tail(self, name: str, n: int = 2000) -> str:
        try:
            return self._log(name).read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def start(self, model: str | None, vector: str | None, scale: float, layers: list[int] | None) -> dict:
        mp = self.model_path(model)
        cmd = [self.server_bin, "-m", str(mp), "--host", self.server_host, "--port", str(self.server_port),
               *self.server_args]
        vp = None
        if vector and scale:
            vp = self.vector_path(mp, vector)
            if not vp.is_file():
                raise AgentFailure(f"no vector '{vector}' for {mp.name}; build it first")
            cmd += ["--control-vector-scaled", str(vp), str(float(scale))]
            if layers:
                cmd += ["--control-vector-layer-range", str(int(layers[0])), str(int(layers[1]))]
        self.stop()
        with open(self._log("server"), "w") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        self.state = {"model": mp.name, "vector": vector if vp else None, "scale": float(scale) if vp else 0.0,
                      "layers": layers if vp else None, "pid": self.proc.pid, "started": time.time()}
        self._wait_healthy()
        return dict(self.state)

    def _wait_healthy(self):
        deadline = time.time() + self.health_timeout
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                self.state = {}
                raise AgentFailure("llama-server exited while starting:\n" + self._tail("server"))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.server_port}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1.0)
        self.stop()
        raise AgentFailure(f"llama-server not healthy after {self.health_timeout:.0f}s:\n" + self._tail("server"))

    def stop(self):
        p, self.proc, self.state = self.proc, None, {}
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)
        except ProcessLookupError:
            pass

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.server_port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    # ---- building vectors -------------------------------------------------------
    def build(self, job: dict, model: str | None, name: str, pairs: list[dict], method: str):
        """Render each prompt pair with the model's chat template, then run llama-cvector-generator."""
        try:
            mp = self.model_path(model)
            out = self.vector_path(mp, name)
            job["status"] = "rendering prompts"
            self.start(model, None, 0.0, None)
            pos, neg = [], []
            for p in pairs:
                suffix = p.get("suffix", "")
                for side, dest in (("positive", pos), ("negative", neg)):
                    prompt = self._post("/apply-template", {"messages": p[side]})["prompt"] + suffix
                    dest.append(prompt.replace("\n", "\\n"))  # the generator reads one prompt per line
            self.stop()  # free the GPU for the generator
            work = self.data_dir / "build" / mp.stem
            work.mkdir(parents=True, exist_ok=True)
            (work / f"{name}.positive.txt").write_text("\n".join(pos) + "\n")
            (work / f"{name}.negative.txt").write_text("\n".join(neg) + "\n")
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [self.gen_bin, "-m", str(mp), "--positive-file", str(work / f"{name}.positive.txt"),
                   "--negative-file", str(work / f"{name}.negative.txt"), "-o", str(out), "--method", method,
                   *self.gen_args]
            job["status"] = f"running llama-cvector-generator on {len(pairs)} pairs"
            with open(self._log("generator"), "w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            if rc != 0 or not out.is_file():
                raise AgentFailure(f"llama-cvector-generator failed (exit {rc}):\n" + self._tail("generator"))
            out.with_suffix(".json").write_text(json.dumps({"model": mp.name, "method": method, "pairs": len(pairs)}))
            job.update(status="done", vector=name, model=mp.name)
        except Exception as e:  # noqa: BLE001 - reported to the client through the job
            self.stop()
            job.update(status="error", error=str(e))
        finally:
            self.lock.release()


# --------------------------------------------------------------------------- HTTP API
class BuildReq(BaseModel):
    name: str
    model: str | None = None
    pairs: list[dict]
    method: str = "pca"


class ServeReq(BaseModel):
    model: str | None = None
    vector: str | None = None
    scale: float = 0.0
    layers: list[int] | None = None


def create_agent_app(agent: Agent, token: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        agent.stop()  # don't leave a llama-server holding the GPU

    app = FastAPI(title="tealeaf steer-agent", lifespan=lifespan)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
            return JSONResponse({"detail": "bad or missing token"}, status_code=401)
        return await call_next(request)

    def busy():
        if not agent.lock.acquire(blocking=False):
            raise HTTPException(409, "the agent is busy (a server start or vector build is in progress)")

    @app.get("/status")
    def status():
        running = agent.proc is not None and agent.proc.poll() is None
        return {"server": {**agent.state, "running": running, "port": agent.server_port},
                "default_model": agent.default_model, "models": agent.models(), "vectors": agent.vectors(),
                "busy": agent.lock.locked()}

    @app.post("/vectors")
    def build(req: BuildReq):
        if req.method not in ("pca", "mean"):
            raise HTTPException(400, "method must be pca or mean")
        if not req.pairs or any(not isinstance(p.get("positive"), list) or not isinstance(p.get("negative"), list)
                                for p in req.pairs):
            raise HTTPException(400, "pairs need 'positive' and 'negative' message lists")
        busy()
        job = {"id": uuid.uuid4().hex[:12], "status": "queued", "started": time.time()}
        agent.jobs[job["id"]] = job
        threading.Thread(target=agent.build, args=(job, req.model, req.name, req.pairs, req.method),
                         daemon=True).start()  # releases the lock when done
        return job

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in agent.jobs:
            raise HTTPException(404)
        return agent.jobs[job_id]

    @app.post("/server")
    def serve(req: ServeReq, request: Request):
        busy()
        try:
            state = agent.start(req.model, req.vector, req.scale, req.layers)
        except AgentFailure as e:
            raise HTTPException(400, str(e))
        finally:
            agent.lock.release()
        host = request.url.hostname or "127.0.0.1"
        host = f"[{host}]" if ":" in host else host
        return {**state, "base_url": f"http://{host}:{agent.server_port}/v1"}

    @app.post("/server/stop")
    def stop():
        busy()
        try:
            agent.stop()
        finally:
            agent.lock.release()
        return {"ok": True}

    return app
