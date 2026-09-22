"""Local review API for the webapp. One model, one job at a time.

    venv-mlx/bin/python inference/server.py        # http://127.0.0.1:8765

POST /api/extract          raw file body, X-Filename header -> {text}
POST /api/reviews          {text, documentName, tasks}      -> {id}
GET  /api/reviews/<id>     -> {status, stage, done, total, window, windows, review?, error?}
GET  /api/health           -> {model: loading|ready|failed, queued}
"""

import json
import queue
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from engine import Engine
from extract import extract_text
from pipeline import Reviewer

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTER = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "adapters" / "llama_3task_mlx")
PORT = 8765
MAX_UPLOAD = 20 * 1024 * 1024
TASKS = {"compliance", "clauses"}

jobs = {}
work = queue.Queue()
state = {"model": "loading"}  # loading | ready | failed: <why>


def worker():
    try:
        reviewer = Reviewer(Engine(MODEL, ADAPTER))
    except Exception as e:
        traceback.print_exc()
        state["model"] = f"failed: {e}"
        return
    state["model"] = "ready"
    print(f"model ready on http://127.0.0.1:{PORT}", flush=True)
    while True:
        jid = work.get()
        job = jobs[jid]
        job["status"] = "running"
        try:
            review = reviewer.run(job.pop("text"), tasks=job["tasks"],
                                  document_name=job["documentName"],
                                  progress=lambda **kw: job.update(kw))
            review["id"] = jid
            job.update(status="done", review=review)
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", error=str(e))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("file too large")
        return self.rfile.read(n)

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, {"model": state["model"], "queued": work.qsize()})
        if self.path.startswith("/api/reviews/"):
            job = jobs.get(self.path.rsplit("/", 1)[1])
            if not job:
                return self._send(404, {"error": "no such review"})
            return self._send(200, {k: v for k, v in job.items() if k != "text"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/extract":
                name = unquote(self.headers.get("X-Filename") or "upload.txt")
                return self._send(200, {"text": extract_text(name, self._body())})
            if self.path == "/api/reviews":
                req = json.loads(self._body() or b"{}")
                text = (req.get("text") or "").strip()
                tasks = [t for t in req.get("tasks") or [] if t in TASKS]
                if not text or not tasks:
                    return self._send(400, {"error": "text and at least one task are required"})
                jid = uuid.uuid4().hex[:10]
                jobs[jid] = {"status": "queued", "stage": "queued", "done": 0, "total": 0,
                             "tasks": tasks, "text": text,
                             "documentName": req.get("documentName") or "Contract"}
                work.put(jid)
                return self._send(200, {"id": jid})
            self._send(404, {"error": "not found"})
        except ValueError as e:
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        if "/api/reviews/" not in self.path:  # don't log every progress poll
            super().log_message(fmt, *args)


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
