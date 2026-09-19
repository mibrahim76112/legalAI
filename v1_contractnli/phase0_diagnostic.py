#!/usr/bin/env python3
"""
PHASE 0 diagnostic. Analysis only -- no training, no GPU.

Runs on the DEV split. Test is reserved and is touched exactly once at the end
of the project, on the winning configuration only.

Does NOT modify evidence_metrics or evidence_case. The failure taxonomy (0.3)
is a NEW measurement built from string operations on the consolidated
text_norm.norm; the existing metrics are imported and used unchanged.
"""
import json, csv, glob, argparse
from pathlib import Path
from collections import Counter, defaultdict

from text_norm import norm
from doc_harness import evidence_case, covered_flags
from doc_metrics import verdict_metrics, evidence_metrics, UNPARSED

CL = ["Entailment", "Contradiction", "NotMentioned"]


def load(fp):
    r = [json.loads(l) for l in open(fp, encoding="utf-8")]
    r.sort(key=lambda x: (x["doc_id"], x["hypothesis_id"]))
    return r


def contracts_for(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        x = json.loads(l)
        u = [m["content"] for m in x["messages"] if m["role"] == "user"][0]
        out[x["doc_id"]] = u.split("CONTRACT: ", 1)[1].rsplit("\n\nQUESTION: ", 1)[0]
    return out


# ---- 0.3 failure taxonomy (NEW measurement, string ops only) --------------
def failure_type(row, contract):
    """Ordered so each failing row lands in exactly one bucket."""
    gold, pred = row["gold_evidence"], row["parsed_evidence"]
    if not pred and gold:
        return "a_missed"
    if pred and not gold:
        return "b_over_extraction"
    hay = norm(contract)
    grounded = []
    for s in pred:
        t = norm(s)
        if t and (t in hay or t.strip(".,;:!?'\"-—’ ") in hay):
            grounded.append(s)
    if len(grounded) < len(pred):
        return "c_fabricated"
    flags = covered_flags(gold, pred)
    return "d_partial" if any(flags) else "e_wrong_span"


def recovered_fraction(row):
    """gold spans recovered / gold spans required."""
    gold, pred = row["gold_evidence"], row["parsed_evidence"]
    if not gold:
        return None
    f = covered_flags(gold, pred)
    return sum(f) / len(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Arm A dev prediction jsonl")
    ap.add_argument("--data", default="doc_sft/valid.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="phase0")
    args = ap.parse_args()

    rows = load(args.pred)
    contracts = contracts_for(args.data)
    hyp_text = json.load(open("data/hypotheses.json", encoding="utf-8")) \
        if Path("data/hypotheses.json").exists() else {}
    outd = Path(args.out); outd.mkdir(exist_ok=True)
    L = []; A = L.append
    A(f"# Phase 0 diagnostic — {args.split} split\n")
    A(f"Predictions: `{Path(args.pred).name}`  ({len(rows)} rows)\n")
    A("Analysis only. **Dev split** — test is reserved and untouched.\n")

    # ---------- 0.1 per-hypothesis ----------
    byh = defaultdict(list)
    for r in rows:
        byh[r["hypothesis_id"]].append(r)
    rec = []
    for h, rs in sorted(byh.items()):
        g = [x["gold_verdict"] for x in rs]
        p = [x["parsed_verdict"] or UNPARSED for x in rs]
        v = verdict_metrics(p, g)
        ev = evidence_metrics(rs)
        exact = sum(1 for x in rs
                    if evidence_case(x["gold_evidence"], x["parsed_evidence"])[0]
                    in ("TP", "TN")) / len(rs)
        d = {"hypothesis": h, "n": len(rs),
             **{f"gold_{c}": sum(1 for y in g if y == c) for c in CL},
             **{f"{c}_{k}": round(v[c][k], 4) for c in CL
                for k in ("precision", "recall", "f1")},
             "macro_f1": round(v["macro_f1"], 4),
             "accuracy": round(v["accuracy"], 4),
             "evidence_exact_rate": round(exact, 4),
             "evidence_f1": round(ev["f1_strict"], 4),
             "evidence_partial_credit": round(ev["partial_credit"], 4)}
        rec.append(d)
    with open(outd / f"per_hypothesis_{args.split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0])); w.writeheader(); w.writerows(rec)
    A("\n## 0.1 Per-hypothesis breakdown\n")
    A("| hyp | n | gold E/C/NM | macro-F1 | acc | ev F1 | ev exact | ev partial |")
    A("|---|---|---|---|---|---|---|---|")
    for d in sorted(rec, key=lambda x: x["evidence_f1"]):
        A(f"| {d['hypothesis']} | {d['n']} | "
          f"{d['gold_Entailment']}/{d['gold_Contradiction']}/{d['gold_NotMentioned']} | "
          f"{d['macro_f1']:.3f} | {d['accuracy']:.3f} | **{d['evidence_f1']:.3f}** | "
          f"{d['evidence_exact_rate']:.3f} | {d['evidence_partial_credit']:.3f} |")

    # ---------- 0.3 failure taxonomy ----------
    fails = [r for r in rows
             if evidence_case(r["gold_evidence"], r["parsed_evidence"])[0] in ("FN", "FP")]
    tax = Counter(); tax_h = defaultdict(Counter)
    for r in fails:
        t = failure_type(r, contracts.get(r["doc_id"], ""))
        tax[t] += 1; tax_h[r["hypothesis_id"]][t] += 1
    nf = sum(tax.values())
    A(f"\n## 0.3 Evidence-failure attribution\n")
    A(f"{nf} failures of {len(rows)} rows ({nf/len(rows):.1%}).\n")
    A("| category | n | % of failures | % of all rows |")
    A("|---|---|---|---|")
    LABEL = {"a_missed": "(a) missed — pred empty, gold non-empty",
             "b_over_extraction": "(b) over-extraction — pred non-empty, gold empty",
             "c_fabricated": "(c) fabricated — span absent from contract",
             "d_partial": "(d) partial — grounded, partial overlap with gold",
             "e_wrong_span": "(e) wrong span — grounded, no overlap"}
    for k in ["a_missed", "b_over_extraction", "c_fabricated", "d_partial", "e_wrong_span"]:
        A(f"| {LABEL[k]} | {tax[k]} | {tax[k]/nf:.1%} | {tax[k]/len(rows):.1%} |")

    # ---------- item 8: recovered-fraction distribution ----------
    fr = [recovered_fraction(r) for r in fails]
    fr = sorted(x for x in fr if x is not None)
    A("\n## 0.3b How near are the near-misses? (item 8)\n")
    if fr:
        q = lambda p: fr[min(int(p*len(fr)), len(fr)-1)]
        A(f"Gold spans recovered / required, over {len(fr)} gold-bearing failures:\n")
        A(f"| p10 | p25 | p50 | p75 | p90 | share ≥0.5 | share =0 |")
        A(f"|---|---|---|---|---|---|---|")
        A(f"| {q(.10):.2f} | {q(.25):.2f} | {q(.50):.2f} | {q(.75):.2f} | {q(.90):.2f} | "
          f"{sum(1 for x in fr if x>=.5)/len(fr):.1%} | {sum(1 for x in fr if x==0)/len(fr):.1%} |")
    ev_all = evidence_metrics(rows)
    A(f"\nStrict evidence F1 (**primary, unchanged**): **{ev_all['f1_strict']:.4f}**")
    A(f"\nPartial-credit variant (reported alongside, NOT a substitute): "
      f"{ev_all['partial_credit']:.4f}\n")

    # ---------- 0.4 positional ----------
    A("\n## 0.4 Positional check\n")
    pos_h = {}
    for h, rs in sorted(byh.items()):
        fracs = []
        for r in rs:
            c = contracts.get(r["doc_id"], "")
            for g in r["gold_evidence"]:
                i = c.find(g[:100])
                if i >= 0 and c:
                    fracs.append(i/len(c))
        if fracs:
            pos_h[h] = (sum(1 for x in fracs if x < .5)/len(fracs), len(fracs))
    allf = [v[0] for v in pos_h.values()]
    A(f"Gold-evidence first-half share, per hypothesis "
      f"(min {min(allf):.1%}, max {max(allf):.1%}):\n")
    A("| hyp | first-half share | n spans | ev F1 |")
    A("|---|---|---|---|")
    evf = {d["hypothesis"]: d["evidence_f1"] for d in rec}
    for h, (s, n) in sorted(pos_h.items(), key=lambda kv: -kv[1][0]):
        A(f"| {h} | {s:.1%} | {n} | {evf.get(h,0):.3f} |")

    # ---------- 0.5 decision ----------
    worst = sorted(rec, key=lambda x: x["evidence_f1"])[:max(1, len(rec)//4)]
    wset = {d["hypothesis"] for d in worst}
    de_worst = sum(tax_h[h]["d_partial"]+tax_h[h]["e_wrong_span"] for h in wset)
    f_all = sum(tax_h[h][k] for h in tax_h for k in tax_h[h])
    ca = (tax["c_fabricated"]+tax["a_missed"])/nf
    de_share_worst = de_worst/nf
    A("\n## 0.5 Decision rule\n")
    A(f"- worst quartile by evidence F1 ({len(wset)} hypotheses): {sorted(wset)}")
    A(f"- (c) fabricated + (a) missed = **{ca:.1%}** of all failures  (threshold ≥40% → PROCEED)")
    A(f"- (d) partial + (e) wrong span **within the worst quartile** = "
      f"**{de_share_worst:.1%}** of all failures")
    branch = "PROCEED to Phase 1" if ca >= 0.40 else "SKIP Phases 1 and 2"
    A(f"\n### BRANCH TAKEN: **{branch}**\n")
    json.dump({"split": args.split, "pred_file": args.pred, "n_rows": len(rows),
               "n_failures": nf, "taxonomy": dict(tax),
               "ca_share": ca, "de_share_worst_quartile": de_share_worst,
               "worst_quartile": sorted(wset), "branch": branch,
               "evidence_f1_strict": ev_all["f1_strict"],
               "evidence_partial_credit": ev_all["partial_credit"]},
              open(outd / f"decision_{args.split}.json", "w"), indent=2)
    (outd / f"PHASE0_{args.split}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-14:]))
    print(f"\n-> {outd}/PHASE0_{args.split}.md, per_hypothesis_{args.split}.csv, "
          f"decision_{args.split}.json")


if __name__ == "__main__":
    main()
