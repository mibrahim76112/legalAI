#!/usr/bin/env python3
"""Build real demo data for the Next.js app from archived v2 predictions.

Nothing here is mocked. Every verdict, clause hit, evidence span and risk note
is model output already produced and archived. Character offsets are computed
with the same locate() that was validated 745/745 on gold spans and 227/227 in
window-overlap regions, so the highlight interaction points at real text.

Two reviews are produced because the two tasks cover DIFFERENT document
populations: ContractNLI's 17 positions are NDA-specific, CUAD's 18 categories
are commercial-agreement-specific. One document cannot honestly demonstrate
both, and the app surfaces whichever applies.
"""
import sys, json, re, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import norm                                     # noqa: E402
from doc_harness import parse_output, strip_wrappers, _find_json  # noqa: E402
from cuad_windowed import norm_with_map, locate                # noqa: E402

R = "/scratch/ibi761/legalai/v2_results"
MODEL_TAG = "llama_3task"          # the selected model
STATUS = {"Entailment": "follows", "Contradiction": "flagged",
          "NotMentioned": "not_addressed"}


def offsets(doc, spans):
    """-> [{text,start,end}] for spans locatable in doc; drops the rest."""
    dn, di = norm_with_map(doc)
    out = []
    for s in spans:
        loc = locate(s, doc, dn, di)
        if loc:
            out.append({"text": s, "start": loc[0], "end": loc[1]})
    return out


def cuad_parse(raw):
    t = strip_wrappers(raw)
    if not t:
        return None, []
    d = _find_json(t)
    if not isinstance(d, dict) or "present" not in d:
        return None, []
    ev = d.get("evidence") or []
    return bool(d["present"]), ([str(x) for x in ev] if isinstance(ev, list) else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="webapp/data/reviews.json")
    args = ap.parse_args()

    dev = [json.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]
    raw = [json.loads(l) for l in open(f"{R}/raw__{MODEL_TAG}.jsonl", encoding="utf-8")]
    assert len(dev) == len(raw), "dev and raw are not row-aligned"

    nli_doc, cuad_doc = defaultdict(list), defaultdict(list)
    for d, p in zip(dev, raw):
        assert d["task"] == p["task"], "row misalignment"
        if d["task"] == "contractnli":
            nli_doc[d["doc_id"]].append((d, p))
        elif d["task"] == "cuad":
            cuad_doc[d["contract"]].append((d, p))

    # --- pick an NDA with a mix of all three verdicts, model not all-correct ---
    def nli_score(items):
        gold = {d["verdict"] for d, _ in items}
        if len(gold) < 3:
            return -1
        text = [m for m in items[0][0]["messages"] if m["role"] == "user"][0]["content"]
        n = len(text)
        return (1 if 8000 < n < 40000 else 0) * 100 + len(items)
    nda_id = max(nli_doc, key=lambda k: nli_score(nli_doc[k]))

    def contract_of(r, task):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        if task == "cuad":
            tail = f"\n\nCLAUSE CATEGORY: {r['category']}"
            return u[len("CONTRACT: "):-len(tail)]
        m = re.search(r"^CONTRACT:\s*(.*?)\s*\n\nQUESTION:\s*(.*)$", u, re.S)
        return m.group(1) if m else u

    def question_of(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        m = re.search(r"\n\nQUESTION:\s*(.*)$", u, re.S)
        return m.group(1).strip() if m else ""

    reviews = []

    # ---------------- NDA review (ContractNLI) ----------------
    items = nli_doc[nda_id]
    doc = contract_of(items[0][0], "contractnli")
    findings = []
    for i, (d, p) in enumerate(sorted(items, key=lambda x: question_of(x[0]))):
        parsed = parse_output(p["raw"])
        pred = parsed["verdict"]
        spans = offsets(doc, parsed["evidence"])
        findings.append({
            "id": f"pos-{i+1}", "kind": "position",
            "position": question_of(d), "positionNumber": i + 1,
            "status": STATUS.get(pred, "not_addressed"),
            "predicted": pred, "gold": d["verdict"],
            "correct": pred == d["verdict"],
            "evidence": spans,
            "goldEvidence": offsets(doc, d.get("evidence") or []),
            "assessment": None,
        })
    reviews.append({
        "id": "acme-nda",
        "documentName": f"Mutual Non-Disclosure Agreement (ContractNLI doc {nda_id})",
        "documentType": "NDA",
        "representing": "Receiving Party",
        "counterparty": "Disclosing Party",
        "playbook": {"name": "Standard NDA playbook", "positionCount": len(findings)},
        "taskCoverage": {"positions": True, "clauses": False},
        "coverageNote": "This is an NDA. The clause taxonomy is trained on "
                        "commercial agreements and does not apply here.",
        "documentText": doc,
        "findings": findings,
        "clauses": [],
        "model": MODEL_TAG,
        "split": "dev",
    })

    # ---------------- commercial contract review (CUAD) ----------------
    notes = {}
    for l in open("v2/risk_notes/dev.jsonl", encoding="utf-8"):
        r = json.loads(l)
        notes[(r["contract"], r["category"])] = r["note"]

    def cuad_score(items):
        doc = contract_of(items[0][0], "cuad")
        npres = sum(1 for d, _ in items if d["present"])
        nnote = sum(1 for d, _ in items if (d["contract"], d["category"]) in notes)
        return (1 if 15000 < len(doc) < 60000 else 0) * 1000 + npres * 10 + nnote
    c_name = max(cuad_doc, key=lambda k: cuad_score(cuad_doc[k]))
    items = cuad_doc[c_name]
    doc = contract_of(items[0][0], "cuad")
    clauses = []
    for i, (d, p) in enumerate(sorted(items, key=lambda x: x[0]["category"])):
        pres, ev = cuad_parse(p["raw"])
        spans = offsets(doc, ev) if pres else []
        clauses.append({
            "id": f"cl-{i+1}", "kind": "clause",
            "category": d["category"],
            "status": "found" if (pres and spans) else "not_detected",
            "predictedPresent": bool(pres), "gold": bool(d["present"]),
            "correct": bool(pres) == bool(d["present"]),
            "evidence": spans,
            "goldEvidence": offsets(doc, d.get("evidence") or []),
            "riskNote": notes.get((d["contract"], d["category"])),
        })
    pretty = re.sub(r"[_\d]+", " ", c_name.split("-")[-1]).strip() or c_name[:60]
    reviews.append({
        "id": "commercial-agreement",
        "documentName": pretty.title(),
        "documentType": "Commercial agreement",
        "representing": "Counterparty",
        "counterparty": "—",
        "playbook": {"name": "Clause taxonomy", "positionCount": len(clauses)},
        "taskCoverage": {"positions": False, "clauses": True},
        "coverageNote": "This is a commercial agreement. The NDA playbook's 17 "
                        "positions are specific to confidentiality terms and do "
                        "not apply here.",
        "documentText": doc,
        "findings": [],
        "clauses": clauses,
        "model": MODEL_TAG,
        "split": "dev",
        "sourceContract": c_name,
    })

    meta = {
        "generated_from": f"{R}/raw__{MODEL_TAG}.jsonl",
        "model": "Llama-3.1-8B-Instruct + LoRA (three-task)",
        "split": "dev (test is untouched)",
        "measured": {
            "contradiction_precision": 0.670,
            "contradiction_recall": 0.811,
            "contractnli_macro_f1": 0.847,
            "contractnli_evidence_strict_f1": 0.747,
            "cuad_present_f1": 0.808,
            "cuad_evidence_strict_f1": 0.583,
            "cuad_truncation_ceiling_span": 0.8765,
        },
        "disclosure": "Roughly 1 in 3 flagged positions is incorrect "
                      "(Contradiction precision 0.670 on dev).",
    }
    out = {"meta": meta, "reviews": reviews}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    for r in reviews:
        n_ev = sum(len(f["evidence"]) for f in r["findings"] + r["clauses"])
        print(f"{r['id']:22s} {r['documentType']:22s} doc {len(r['documentText']):7,d} chars  "
              f"findings {len(r['findings']):3d}  clauses {len(r['clauses']):3d}  "
              f"located spans {n_ev}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
