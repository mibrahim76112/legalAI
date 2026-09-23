"""Local review API for the webapp. One model, one review at a time.

    venv-mlx/bin/python inference/server.py        # http://127.0.0.1:8765

Reviews and decisions persist in inference/store/ (see store.py); live
progress of the running review is kept in memory only.

GET    /api/health                      -> {model: loading|ready|failed: ..., queued}
POST   /api/extract                     raw file body, X-Filename header -> {text}
GET    /api/reviews                     -> [summary]
POST   /api/reviews                     {text, documentName, tasks} -> {id}
GET    /api/reviews/<id>                -> review record + progress
DELETE /api/reviews/<id>
POST   /api/reviews/<id>/rerun          re-queue with the stored text
PUT    /api/reviews/<id>/items/<item>   {decision: ok|flag|skip|null, comment}
"""

import json
import queue
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import store
from engine import Engine
from extract import extract_text
from pipeline import Reviewer
from remote import RemoteEngine, load_env

# local (MLX on this Mac) or remote (vLLM endpoint); see deploy/README.md
BACKEND = load_env("LEGALAI_BACKEND", "local")
MODEL = load_env("LEGALAI_LOCAL_MODEL", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
ADAPTER = load_env("LEGALAI_LOCAL_ADAPTER",
                   str(Path(__file__).resolve().parent.parent / "adapters" / "llama_3task_mlx"))
PORT = 8765
MAX_UPLOAD = 20 * 1024 * 1024
TASKS = ("compliance", "clauses")

work = queue.Queue()
progress = {}  # review id -> latest progress of the running review
state = {"model": "loading"}  # loading | ready | failed: <why>


def reset_results(rec):
    rec.update(status="queued", compliance=[], clauses=[], stats=None, error=None)


def worker():
    try:
        reviewer = Reviewer(RemoteEngine() if BACKEND == "remote" else Engine(MODEL, ADAPTER))
    except Exception as e:
        traceback.print_exc()
        state["model"] = f"failed: {e}"
        return
    state["model"] = "ready"
    where = f"remote {reviewer.eng.model}" if BACKEND == "remote" else f"local {MODEL}"
    print(f"model ready ({where}) on http://127.0.0.1:{PORT}", flush=True)
    while True:
        rid = work.get()
        rec = store.update(rid, lambda r: r.update(status="running"))
        if rec is None:  # deleted while queued
            continue
        progress[rid] = {"stage": "queued", "done": 0, "total": 0}
        try:
            out = reviewer.run(rec["documentText"], tasks=rec["tasks"],
                               document_name=rec["documentName"],
                               progress=lambda **kw: progress[rid].update(kw))
            keep = ("compliance", "clauses", "stats", "representing", "counterparty", "playbookName")
            store.update(rid, lambda r: r.update(status="done", **{k: out[k] for k in keep}))
        except Exception as e:
            traceback.print_exc()
            store.update(rid, lambda r: r.update(status="error", error=str(e)))
        finally:
            progress.pop(rid, None)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj=None):
        body = json.dumps(obj if obj is not None else {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("file too large (20 MB max)")
        return self.rfile.read(n)

    def _json(self):
        return json.loads(self._body() or b"{}")

    def _parts(self):
        return [p for p in urlparse(self.path).path.split("/") if p][1:]  # drop "api"

    def do_GET(self):
        p = self._parts()
        if p == ["health"]:
            return self._send(200, {"model": state["model"], "queued": work.qsize()})
        if p == ["reviews"]:
            return self._send(200, store.list_summaries())
        if len(p) == 2 and p[0] == "reviews":
            rec = store.load(p[1])
            if rec is None:
                return self._send(404, {"error": "no such review"})
            return self._send(200, {**rec, "progress": progress.get(p[1])})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        p = self._parts()
        try:
            if p == ["extract"]:
                name = unquote(self.headers.get("X-Filename") or "upload.txt")
                return self._send(200, {"text": extract_text(name, self._body())})
            if p == ["reviews"]:
                req = self._json()
                text = (req.get("text") or "").strip()
                tasks = [t for t in TASKS if t in (req.get("tasks") or [])]
                if not text or not tasks:
                    return self._send(400, {"error": "text and at least one check are required"})
                rec = {"id": uuid.uuid4().hex[:10], "createdAt": time.time(), "tasks": tasks,
                       "documentName": req.get("documentName") or "Contract",
                       "documentText": text, "sample": False,
                       "representing": None, "counterparty": None, "playbookName": None}
                reset_results(rec)
                store.save(rec)
                work.put(rec["id"])
                return self._send(200, {"id": rec["id"]})
            if len(p) == 3 and p[0] == "reviews" and p[2] == "rerun":
                rec = store.load(p[1])
                if rec is None:
                    return self._send(404, {"error": "no such review"})
                if rec.get("sample") or rec["status"] in ("queued", "running"):
                    return self._send(409, {"error": "this review cannot be re-run now"})
                store.update(p[1], reset_results)
                work.put(p[1])
                return self._send(200, {"id": p[1]})
            self._send(404, {"error": "not found"})
        except ValueError as e:
            self._send(400, {"error": str(e)})

    def do_PUT(self):
        p = self._parts()
        if len(p) == 4 and p[0] == "reviews" and p[2] == "items":
            try:
                req = self._json()
                rec = store.set_decision(p[1], p[3], req.get("decision"), req.get("comment"))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except KeyError:
                return self._send(404, {"error": "no such finding"})
            if rec is None:
                return self._send(404, {"error": "no such review"})
            return self._send(200, store.summary(rec))
        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        p = self._parts()
        if len(p) == 2 and p[0] == "reviews":
            if progress.get(p[1]):
                return self._send(409, {"error": "wait for the review to finish first"})
            return self._send(200 if store.delete(p[1]) else 404)
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        if self.command != "GET":  # skip progress polls
            super().log_message(fmt, *args)


if __name__ == "__main__":
    store.seed_samples()
    store.mark_interrupted()
    threading.Thread(target=worker, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
