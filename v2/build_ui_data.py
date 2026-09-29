#!/usr/bin/env python3
"""Build the UI payload: ONE review, both tabs, same document.

Prefers webapp/data/one_document.json (both task heads run over a single
contract). Falls back to the archived two-review data only if that is absent.
"""
import sys, json, re, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import norm                                     # noqa: E402
from cuad_windowed import norm_with_map, locate                # noqa: E402

TONE = {"Entailment": ("grn", "Meets standard"),
        "Contradiction": ("red", "Needs attention"),
        "NotMentioned": ("amb", "Not addressed")}
ASSESS = {"Entailment": "The contract supports this position.",
          "Contradiction": "The contract appears to conflict with this position.",
          "NotMentioned": "The contract does not address this."}


def spans_of(doc, texts):
    dn, di = norm_with_map(doc)
    out = []
    for t in texts:
        loc = locate(t, doc, dn, di)
        if loc:
            out.append({"text": t, "start": loc[0], "end": loc[1]})
    return out


def from_archive(out_path):
    """Build reviews from archived predictions -- no GPU, no fabrication."""
    import json as _j
    from collections import Counter
    from doc_harness import parse_output, strip_wrappers, _find_json
    R = "/scratch/ibi761/legalai/v2_results"
    dev = [_j.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]
    raw = [_j.loads(l) for l in open(f"{R}/raw__llama_3task.jsonl", encoding="utf-8")]
    notes = {}
    for l in open("v2/risk_notes/dev.jsonl", encoding="utf-8"):
        r = _j.loads(l); notes[(r["contract"], r["category"])] = r["note"]

    nli, cuad = {}, {}
    for d0, p0 in zip(dev, raw):
        if d0["task"] == "contractnli":
            nli.setdefault(d0["doc_id"], []).append((d0, p0))
        elif d0["task"] == "cuad":
            cuad.setdefault(d0["contract"], []).append((d0, p0))

    def q_of(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        m = re.search(r"\n\nQUESTION:\s*(.*)$", u, re.S)
        return m.group(1).strip() if m else ""

    def doc_nli(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        m = re.search(r"^CONTRACT:\s*(.*?)\s*\n\nQUESTION:", u, re.S)
        return m.group(1) if m else u

    def doc_cuad(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        tail = f"\n\nCLAUSE CATEGORY: {r['category']}"
        return u[len("CONTRACT: "):-len(tail)]

    def cparse(rw):
        t = strip_wrappers(rw)
        j = _find_json(t) if t else None
        if not isinstance(j, dict) or "present" not in j:
            return False, []
        ev = j.get("evidence") or []
        return bool(j["present"]), ([str(x) for x in ev] if isinstance(ev, list) else [])

    reviews = []
    # NDA -> compliance tab
    def score(k):
        it = nli[k]
        if len({a["verdict"] for a, _ in it}) < 3: return -1
        n = len(doc_nli(it[0][0]))
        return (1 if 8000 < n < 40000 else 0) * 100 + len(it)
    k = max(nli, key=score)
    it = sorted(nli[k], key=lambda x: q_of(x[0]))
    doc = doc_nli(it[0][0])
    comp = []
    for i, (a, b) in enumerate(it):
        pr = parse_output(b["raw"])
        v = pr["verdict"] or "NotMentioned"
        tone, label = TONE.get(v, ("gry", "Unclear"))
        comp.append({"id": f"p{i}", "kind": "position", "title": q_of(a),
                     "status": label, "tone": tone,
                     "evidence": spans_of(doc, pr["evidence"]),
                     "note": ASSESS.get(v), "correct": v == a["verdict"]})
    reviews.append({"id": "nda", "documentName": "Non-Disclosure Agreement",
                    "representing": "Receiving Party",
                    "counterparty": "Disclosing Party",
                    "playbookName": "Standard playbook", "documentText": doc,
                    "compliance": comp, "clauses": []})

    # commercial -> clauses tab
    def cscore(c):
        it = cuad[c]; d0 = doc_cuad(it[0][0])
        return (1 if 15000 < len(d0) < 60000 else 0) * 1000 + sum(
            1 for a, _ in it if a["present"])
    c = max(cuad, key=cscore)
    it = sorted(cuad[c], key=lambda x: x[0]["category"])
    doc2 = doc_cuad(it[0][0])
    cl = []
    for i, (a, b) in enumerate(it):
        pres, ev = cparse(b["raw"])
        sp = spans_of(doc2, ev) if pres else []
        cl.append({"id": f"c{i}", "kind": "clause", "title": a["category"],
                   "status": "Found" if sp else "Not detected",
                   "tone": "grn" if sp else "gry", "evidence": sp,
                   "note": notes.get((a["contract"], a["category"]))})
    nice = re.sub(r"[_\d]+", " ", c.split("-")[-1]).strip().title() or "Commercial Agreement"
    reviews.append({"id": "commercial", "documentName": nice,
                    "representing": "Counterparty", "counterparty": "—",
                    "playbookName": "Standard playbook", "documentText": doc2,
                    "compliance": [], "clauses": cl})

    payload = {"meta": {"model": "Llama-3.1-8B + LoRA (three-task)",
                        "split": "dev", "flagPrecision": 0.670},
               "reviews": reviews}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _j.dump(payload, open(out_path, "w"), indent=1)
    for r in reviews:
        print(f"{r['id']:12s} compliance {len(r['compliance']):2d}  "
              f"clauses {len(r['clauses']):2d}  "
              f"spans {sum(len(i['evidence']) for i in r['compliance']+r['clauses'])}")
    print(f"-> {out_path}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="webapp/data/one_document.json")
    ap.add_argument("--out", default="webapp/data/ui.json")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        # No GPU run yet: assemble from archived predictions instead. Each
        # review then carries whichever tab that contract was actually scored
        # on; the other tab shows a quiet empty state rather than a lecture.
        return from_archive(args.out)
    d = json.load(open(src))
    doc = d["documentText"]
    comp, clauses = [], []
    for k, r in enumerate(d["rows"]):
        if r["kind"] == "position":
            v = r["predicted"] or "NotMentioned"
            tone, label = TONE.get(v, ("gry", "Unclear"))
            comp.append({"id": f"p{k}", "kind": "position",
                         "title": r["position"], "status": label, "tone": tone,
                         "evidence": spans_of(doc, r["evidence"]),
                         "note": ASSESS.get(v),
                         "correct": v == r["gold"]})
        else:
            ev = spans_of(doc, r["evidence"]) if r["predictedPresent"] else []
            found = bool(ev)
            clauses.append({"id": f"c{k}", "kind": "clause",
                            "title": r["category"],
                            "status": "Found" if found else "Not detected",
                            "tone": "grn" if found else "gry",
                            "evidence": ev,
                            "note": r.get("riskNote")})

    out = {
        "meta": {"model": "Qwen3-4B + LoRA (three-task)", "split": "dev",
                 "flagPrecision": 0.623},
        "reviews": [{
            "id": "review-1",
            "documentName": "Non-Disclosure Agreement",
            "representing": "Receiving Party",
            "counterparty": "Disclosing Party",
            "playbookName": "Standard playbook",
            "documentText": doc,
            "compliance": comp,
            "clauses": clauses,
        }],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    from collections import Counter
    print(f"compliance {len(comp)} {dict(Counter(i['tone'] for i in comp))}")
    print(f"clauses    {len(clauses)} {dict(Counter(i['tone'] for i in clauses))}")
    print(f"spans      {sum(len(i['evidence']) for i in comp+clauses)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
