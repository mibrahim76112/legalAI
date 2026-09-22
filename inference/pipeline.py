"""Contract -> Review: the three-task adapter run over one live document.

Contracts longer than the context budget are split into overlapping windows
(the cluster's CUAD path was windowed too, see macbundle/README). Windows are
the outer loop and questions the inner one, so each window's prefill is paid
once and reused for every question through the engine's prefix cache.

Output matches webapp/lib/types.ts `Review`.
"""

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "macbundle" / "lib"))

import prompts as P  # noqa: E402
from doc_harness import parse_output, _find_json, strip_wrappers  # noqa: E402
from spans import norm_with_map, spans_of  # noqa: E402
from text_norm import norm  # noqa: E402

from engine import MAX_CONTEXT, MAX_NEW_TOKENS  # noqa: E402

# The system prompt the task-3 validation rows (and so the cluster's bf16
# output) actually use. prompts.TASK3_SYSTEM is the longer teacher prompt;
# on validation_60 it agreed with the cluster less (token Jaccard 0.39 vs 0.60).
TASK3_SYSTEM_TRAINED = (
    "You are a contract review assistant. Given a clause and its category, write "
    "EXACTLY ONE sentence for a business reader explaining what the clause does in "
    "practice and what brings it into effect."
)

# Room left in the prompt for the system message, the question and the
# chat-template tokens around them. The longest position is ~40 tokens.
PROMPT_OVERHEAD = 400
WINDOW_TOKENS = MAX_CONTEXT - MAX_NEW_TOKENS - PROMPT_OVERHEAD
OVERLAP_TOKENS = 1024
# a window edge moves back to the nearest line break within this many chars
SNAP_CHARS = 400

POSITION_STATUS = {
    "Entailment": ("Meets standard", "grn", "The contract supports this position."),
    "Contradiction": ("Needs attention", "red", "The contract appears to conflict with this position."),
    "NotMentioned": ("Not addressed", "amb", "The contract does not address this."),
}
# Across windows a conflict anywhere outranks support anywhere, which
# outranks silence: a clause only has to appear in one window to count.
VERDICT_RANK = {"Contradiction": 0, "Entailment": 1, "NotMentioned": 2}


def windows(tok, text):
    """-> [(start_char, end_char)] covering text, each within WINDOW_TOKENS."""
    enc = tok._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if len(offs) <= WINDOW_TOKENS:
        return [(0, len(text))]
    out, t = [], 0
    while True:
        start = offs[t][0]
        last = t + WINDOW_TOKENS
        if last >= len(offs):
            out.append((start, len(text)))
            return out
        end = offs[last][0]
        nl = text.rfind("\n", end - SNAP_CHARS, end)
        if nl > start:
            end = nl + 1
        out.append((start, end))
        # next window starts OVERLAP_TOKENS before this one ends
        t = next(i for i, (a, _) in enumerate(offs) if a >= end) - OVERLAP_TOKENS
        nl = text.rfind("\n", offs[t][0] - SNAP_CHARS, offs[t][0])
        if nl > start:
            t = next(i for i, (a, _) in enumerate(offs) if a >= nl + 1)


def parse_cuad(raw):
    obj = _find_json(strip_wrappers(raw))
    if not isinstance(obj, dict) or not isinstance(obj.get("present"), bool):
        return None, []
    ev = obj.get("evidence") or []
    if isinstance(ev, str):
        ev = [ev]
    return obj["present"], [e for e in ev if isinstance(e, str) and e.strip()]


def _dedupe(texts):
    seen, out = set(), []
    for t in texts:
        k = norm(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


class Reviewer:
    def __init__(self, engine):
        self.eng = engine

    def run(self, text, tasks=("compliance", "clauses"), document_name="Contract",
            positions=None, categories=None, progress=lambda **kw: None):
        positions = positions or P.PLAYBOOK_POSITIONS
        categories = categories or P.CLAUSE_CATEGORIES
        wins = windows(self.eng.tok, text)
        dn, di = norm_with_map(text)
        assert dn == norm(text), "norm_with_map drifted from norm"

        qs = []
        if "compliance" in tasks:
            qs += [("nli", q) for q in positions]
        if "clauses" in tasks:
            qs += [("cuad", c) for c in categories]
        total = len(wins) * len(qs)
        done, stats = 0, {"windows": len(wins), "unparsed": 0, "unlocated_quotes": 0}

        per_q = {q: [] for q in qs}  # (label, evidence) per window
        for w, (a, b) in enumerate(wins):
            chunk = text[a:b]
            self.eng.reset_cache()
            for kind, q in qs:
                if kind == "nli":
                    msgs = [{"role": "system", "content": P.TASK1_SYSTEM},
                            {"role": "user", "content": P.TASK1_USER.format(text=chunk, question=q)}]
                    raw = self.eng.generate(msgs, reuse_prefix=True).text
                    p = parse_output(raw)
                    label, ev = p["verdict"], p["evidence"]
                else:
                    msgs = [{"role": "system", "content": P.TASK2_SYSTEM},
                            {"role": "user", "content": P.TASK2_USER.format(text=chunk, category=q)}]
                    raw = self.eng.generate(msgs, reuse_prefix=True).text
                    label, ev = parse_cuad(raw)
                if label is None:
                    stats["unparsed"] += 1
                per_q[(kind, q)].append((label, ev))
                done += 1
                progress(stage="clauses" if kind == "cuad" else "positions",
                         done=done, total=total, window=w + 1, windows=len(wins))
        self.eng.reset_cache()

        def located(ev):
            s = spans_of(text, _dedupe(ev))
            stats["unlocated_quotes"] += len(_dedupe(ev)) - len(s)
            return s

        compliance, clauses = [], []
        for i, q in enumerate(positions if "compliance" in tasks else []):
            res = [r for r in per_q[("nli", q)] if r[0]]
            verdict = min((r[0] for r in res), key=VERDICT_RANK.get, default="NotMentioned")
            ev = [e for v, evs in res if v == verdict for e in evs]
            status, tone, note = POSITION_STATUS[verdict]
            compliance.append({"id": f"p{i}", "kind": "position", "title": q, "status": status,
                               "tone": tone, "evidence": located(ev), "note": note})

        found = []
        for i, c in enumerate(categories if "clauses" in tasks else []):
            res = per_q[("cuad", c)]
            present = any(r[0] for r in res)
            ev = [e for v, evs in res if v for e in evs]
            item = {"id": f"c{i}", "kind": "clause", "title": c,
                    "status": "Found" if present else "Not detected",
                    "tone": "grn" if present else "gry",
                    "evidence": located(ev) if present else [], "note": None}
            clauses.append(item)
            if present and item["evidence"]:
                found.append(item)

        for k, item in enumerate(found):
            progress(stage="notes", done=k, total=len(found))
            clause = " ".join(s["text"] for s in item["evidence"])
            msgs = [{"role": "system", "content": TASK3_SYSTEM_TRAINED},
                    {"role": "user", "content": P.TASK3_USER.format(cat=item["title"], clause=clause)}]
            item["note"] = strip_wrappers(self.eng.generate(msgs).text).strip()

        return {
            "id": uuid.uuid4().hex[:10],
            "documentName": document_name,
            "representing": "Receiving Party" if "compliance" in tasks else "Counterparty",
            "counterparty": "Disclosing Party" if "compliance" in tasks else "—",
            "playbookName": "Standard playbook",
            "documentText": text,
            "compliance": compliance,
            "clauses": clauses,
            "stats": stats,
        }
