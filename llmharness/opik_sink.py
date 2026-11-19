"""Optional: mirror each case to Opik as a trace with feedback scores. Never breaks a run."""
from __future__ import annotations

import datetime as dt
import sys

from .config import OpikSettings


class OpikSink:
    def __init__(self, settings: OpikSettings):
        import opik  # optional dependency: pip install 'llm-test-harness[opik]'

        kw = {"project_name": settings.project, "host": settings.url, "workspace": settings.workspace}
        if settings.api_key:
            kw["api_key"] = settings.api_key
        self.client = opik.Opik(**kw)
        self.project = settings.project
        self._warned = False

    def _warn(self, e: Exception):
        if not self._warned:
            print(f"[opik] logging failed (continuing without it): {e}", file=sys.stderr)
            self._warned = True

    def log_case(self, runner, case, fields: dict, messages: list[dict], t0: float, t1: float):
        try:
            target_msgs = [m for m in messages if m["thread"] != "judge"]
            trace = self.client.trace(
                name=case.id,
                start_time=dt.datetime.fromtimestamp(t0, dt.timezone.utc),
                end_time=dt.datetime.fromtimestamp(t1, dt.timezone.utc),
                input={"case": case.title, "description": case.description,
                       "messages": [{"thread": m["thread"], "role": m["role"], "content": m["content"]}
                                    for m in target_msgs if m["role"] != "assistant"]},
                output={"replies": [m["content"] for m in target_msgs if m["role"] == "assistant"],
                        "judge": fields.get("judge")},
                metadata={"run_id": runner.run_id, "model": runner.target.model, "status": fields["status"],
                          "metrics": fields.get("metrics"), "notes": fields.get("notes"), "error": fields.get("error")},
                tags=["llm-test-harness", case.suite, case.kind, runner.target.model,
                      *[f["name"] for f in fields.get("flags") or []][:10]],
            )
            scores = []
            if fields.get("score") is not None:
                scores.append({"id": trace.id, "name": "score", "value": float(fields["score"])})
            if fields.get("interest") is not None:
                scores.append({"id": trace.id, "name": "interest", "value": float(fields["interest"]) / 10})
            if scores:
                self.client.log_traces_feedback_scores(scores=scores, project_name=self.project)
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def flush(self):
        try:
            self.client.flush()
        except Exception as e:  # noqa: BLE001
            self._warn(e)
