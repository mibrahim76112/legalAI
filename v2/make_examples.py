#!/usr/bin/env python3
"""Pull real task examples for the deck: actual inputs, actual model outputs.

Contracts run to 30k+ characters, so the excerpt is windowed around the gold
evidence with an ellipsis marker. The evidence text itself is never altered.
"""
import json, re, sys, textwrap, argparse
from pathlib import Path
sys.path.insert(0, "v1_contractnli")
from text_norm import norm
from doc_harness import parse_output, strip_wrappers, _find_json

R = "/scratch/ibi761/legalai/v2_results"


def excerpt(doc, spans, width=520):
    """Window around the first gold span so the slide shows the relevant part."""
    if not spans:
        return textwrap.shorten(doc[:width * 2], width, placeholder=" …")
    key = spans[0]
    i = doc.find(key[:60])
    if i < 0:
        n = norm(doc); j = n.find(norm(key)[:60])
        i = max(j, 0)
    lo = max(0, i - width // 3)
    hi = min(len(doc), i + len(key) + width // 3)
    out = doc[lo:hi].strip()
    return ("… " if lo > 0 else "") + out + (" …" if hi < len(doc) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama_3task")
    ap.add_argument("--out", default="v2/TASK_EXAMPLES.md")
    args = ap.parse_args()
    dev = [json.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]
    raw = [json.loads(l) for l in open(f"{R}/raw__{args.tag}.jsonl", encoding="utf-8")]
    assert len(dev) == len(raw)

    def user(r):
        return [m for m in r["messages"] if m["role"] == "user"][0]["content"]

    L = []; W = L.append
    W("# What the three tasks look like\n")
    W(f"Real inputs, real outputs from the fine-tuned model (`{args.tag}`, dev split).")
    W("Contract text is excerpted around the relevant clause; quoted spans are verbatim.\n")

    # ---- Task 1: one example per verdict ----
    W("\n## Task 1 — Compliance check (ContractNLI)\n")
    W("**Input:** the contract + one playbook position. "
      "**Output:** verdict + the language that justifies it.\n")
    seen = set()
    for d, p in zip(dev, raw):
        if d["task"] != "contractnli" or d["verdict"] in seen:
            continue
        pr = parse_output(p["raw"])
        if pr["verdict"] != d["verdict"]:
            continue                      # show correct ones for the walkthrough
        seen.add(d["verdict"])
        u = user(d)
        m = re.search(r"^CONTRACT:\s*(.*?)\s*\n\nQUESTION:\s*(.*)$", u, re.S)
        doc, q = m.group(1), m.group(2).strip()
        W(f"\n### {d['verdict']}\n")
        W(f"**Playbook position:** {q}\n")
        W("**Contract (excerpt):**\n")
        W("```")
        W(textwrap.fill(excerpt(doc, d.get("evidence") or []), 88))
        W("```\n")
        W("**Model output:**\n")
        W("```json")
        W(json.dumps({"verdict": pr["verdict"],
                      "evidence": [e[:160] + ("…" if len(e) > 160 else "")
                                   for e in pr["evidence"]]}, indent=2)[:900])
        W("```")
        if len(seen) == 3:
            break

    # ---- Task 2: one present, one absent ----
    W("\n\n## Task 2 — Clause identification (CUAD)\n")
    W("**Input:** the contract + one clause category. "
      "**Output:** present or not, plus the clause text.\n")
    def cparse(rw):
        t = strip_wrappers(rw); j = _find_json(t) if t else None
        if not isinstance(j, dict) or "present" not in j:
            return None, []
        ev = j.get("evidence") or []
        return bool(j["present"]), [str(x) for x in ev]
    got = set()
    for d, p in zip(dev, raw):
        if d["task"] != "cuad":
            continue
        pres, ev = cparse(p["raw"])
        if pres != bool(d["present"]) or pres in got:
            continue
        if pres and not ev:
            continue
        got.add(pres)
        u = user(d)
        tail = f"\n\nCLAUSE CATEGORY: {d['category']}"
        doc = u[len("CONTRACT: "):-len(tail)]
        W(f"\n### {'Found' if pres else 'Not detected'} — {d['category']}\n")
        W("**Contract (excerpt):**\n")
        W("```")
        W(textwrap.fill(excerpt(doc, d.get("evidence") or []), 88))
        W("```\n")
        W("**Model output:**\n")
        W("```json")
        W(json.dumps({"present": pres,
                      "evidence": [e[:200] + ("…" if len(e) > 200 else "")
                                   for e in ev]}, indent=2)[:900])
        W("```")
        if len(got) == 2:
            break

    # ---- Task 3 ----
    W("\n\n## Task 3 — Risk note (synthetic)\n")
    W("**Input:** one clause + its category. **Output:** one plain-English sentence.\n")
    n = 0
    for d, p in zip(dev, raw):
        if d["task"] != "risknote":
            continue
        u = user(d)
        m = re.search(r"CLAUSE CATEGORY:\s*(.*?)\n\nCLAUSE:\s*(.*)$", u, re.S)
        if not m:
            continue
        W(f"\n### {m.group(1).strip()}\n")
        W("**Clause:**\n")
        W("```")
        W(textwrap.fill(m.group(2).strip()[:560], 88))
        W("```\n")
        W(f"**Model output:** {' '.join(p['raw'].split())[:400]}\n")
        n += 1
        if n == 2:
            break

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"-> {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
