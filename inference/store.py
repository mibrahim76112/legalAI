"""Reviews and reviewer decisions as one JSON file per review.

A record is the webapp's Review plus bookkeeping: status (queued | running |
done | error | interrupted), createdAt, tasks, sample, error. While a review
is queued or running its findings are empty and documentText holds the text
to analyse, which is also what a re-run uses.

Writes go through a temp file and os.replace, so a crash mid-write leaves the
previous version intact.
"""

import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "store" / "reviews"
SAMPLES = ROOT.parent / "webapp" / "data" / "ui.json"
DECISIONS = {"ok", "flag", "skip", None}

_lock = threading.Lock()


def _path(rid):
    if not rid or not all(c.isalnum() or c in "-_" for c in rid):
        raise KeyError(rid)
    return DIR / f"{rid}.json"


def _write(rec):
    DIR.mkdir(parents=True, exist_ok=True)
    p = _path(rec["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, p)


def load(rid):
    try:
        return json.loads(_path(rid).read_text())
    except (FileNotFoundError, KeyError):
        return None


def save(rec):
    with _lock:
        rec["updatedAt"] = time.time()
        _write(rec)


def update(rid, fn):
    """Load, apply fn(rec) in place, save; all under the lock. -> rec or None."""
    with _lock:
        rec = load(rid)
        if rec is None:
            return None
        fn(rec)
        rec["updatedAt"] = time.time()
        _write(rec)
        return rec


def delete(rid):
    with _lock:
        try:
            _path(rid).unlink()
            return True
        except (FileNotFoundError, KeyError):
            return False


def set_decision(rid, item_id, decision, comment):
    if decision not in DECISIONS:
        raise ValueError("decision must be one of ok, flag, skip or null")

    def apply(rec):
        for it in rec["compliance"] + rec["clauses"]:
            if it["id"] == item_id:
                it["decision"] = decision
                it["comment"] = (comment or "").strip() or None
                return
        raise KeyError(item_id)

    return update(rid, apply)


def summary(rec):
    items = rec.get("compliance", []) + rec.get("clauses", [])
    tone = lambda t: sum(1 for i in rec.get("compliance", []) if i["tone"] == t)
    return {
        "id": rec["id"], "documentName": rec["documentName"], "status": rec["status"],
        "createdAt": rec.get("createdAt"), "tasks": rec.get("tasks", []),
        "sample": rec.get("sample", False), "representing": rec.get("representing"),
        "counts": {"red": tone("red"), "amb": tone("amb"), "grn": tone("grn"),
                   "found": sum(1 for i in rec.get("clauses", []) if i["tone"] == "grn")},
        "decided": sum(1 for i in items if i.get("decision")),
        "total": len(items),
    }


def list_summaries():
    DIR.mkdir(parents=True, exist_ok=True)
    recs = (load(p.stem) for p in DIR.glob("*.json"))
    return sorted((summary(r) for r in recs if r), key=lambda s: -(s["createdAt"] or 0))


def mark_interrupted():
    """Anything queued or running when the server stopped will never finish."""
    for p in DIR.glob("*.json"):
        update(p.stem, lambda r: r.update(status="interrupted")
               if r["status"] in ("queued", "running") else None)


def seed_samples():
    """Import the two archived cluster reviews once, as read-only samples."""
    if not SAMPLES.exists():
        return
    data = json.loads(SAMPLES.read_text())
    for i, r in enumerate(data["reviews"]):
        if load(r["id"]):
            continue
        for it in r["compliance"]:
            it["assessment"], it["note"], it["noteSource"] = it["note"], None, None
        for it in r["clauses"]:
            it["noteSource"] = "model" if it["note"] else None
        tasks = [t for t in ("compliance", "clauses") if r[t]]
        save({**r, "status": "done", "tasks": tasks, "sample": True,
              "createdAt": 1_000_000_000 + i, "stats": None})
