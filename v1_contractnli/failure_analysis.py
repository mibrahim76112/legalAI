#!/usr/bin/env python3
"""
Failure analysis on archived Phase 1 outputs. CPU only, no inference.

Condition A uses the 1024-token re-run (the 512-cap versions are superseded and
were cap-suppressed). Condition B is unaffected and used as-is.
"""
import json, glob, re, argparse, random, statistics as st
from pathlib import Path
from collections import Counter, defaultdict
from doc_harness import norm, evidence_case, covered_flags

CARVE = ["except", "unless", "provided that", "other than", "save for", "subject to"]
CARVE_RE = re.compile("|".join(re.escape(c) for c in CARVE), re.I)
CTX = 300


def load(fp):
    r = [json.loads(l) for l in open(fp, encoding="utf-8")]
    r.sort(key=lambda x: (x["doc_id"], x["hypothesis_id"]))
    return r


def cells(results_dir):
    out = {}
    for fp in sorted(glob.glob(str(Path(results_dir) / "raw__*mnt1024*.jsonl"))):
        r = load(fp); out[(r[0]["model"], "A")] = r
    for fp in sorted(glob.glob(str(Path(results_dir) / "raw__*__think__dev.jsonl"))):
        r = load(fp); out[(r[0]["model"], "B")] = r
    return out


def contract_index(path="doc_sft/valid.jsonl"):
    """doc_id -> full contract text, taken from the user turn we actually sent."""
    idx, hyp = {}, {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        u = [m["content"] for m in r["messages"] if m["role"] == "user"][0]
        body = u.split("CONTRACT: ", 1)[1]
        text, question = body.rsplit("\n\nQUESTION: ", 1)
        idx[r["doc_id"]] = text
        hyp[r["hypothesis_id"]] = question
    return idx, hyp


def tok_overlap(a_spans, b_spans):
    """token-set overlap of prediction vs gold (|A n B| / |B|, recall-flavoured)."""
    a = set(norm(" ".join(a_spans)).split())
    b = set(norm(" ".join(b_spans)).split())
    if not b:
        return None
    return len(a & b) / len(b)


STRICT_CARVE = ["except", "unless", "provided that", "other than", "save for"]
STRICT_RE = re.compile("|".join(re.escape(c) for c in STRICT_CARVE), re.I)


def has_carveout(text, gold_spans, strict=False):
    """carve-out language in a gold span or within +/-300 chars of it.

    strict=True drops "subject to": it appears in most commercial contracts as
    boilerplate cross-referencing, not as a genuine exception, so including it
    makes the subset ~2/3 of all rows and dilutes the test.
    """
    rx = STRICT_RE if strict else CARVE_RE
    for g in gold_spans:
        i = text.find(g[:120])
        if i < 0:
            continue
        seg = text[max(0, i - CTX): i + len(g) + CTX]
        if rx.search(seg):
            return True
    return bool(gold_spans) and any(rx.search(g) for g in gold_spans)


def short(s, n=320):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " …"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/scratch/ibi761/legalai/doc_results")
    ap.add_argument("--out", default="FAILURE_ANALYSIS.md")
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C = cells(args.dir)
    text_of, question_of = contract_index()
    rng = random.Random(args.seed)
    L = []; A = L.append

    A("# Failure analysis — Phase 1 base models (pre-SFT)\n")
    A("Archived outputs only, no inference. Condition **A** uses the 1024-token")
    A("re-run; the 512-cap versions were cap-suppressed and are superseded.")
    A("Condition **B** (thinking) is unaffected.\n")
    A(f"Dev split, 1037 (document, hypothesis) pairs per cell, "
      f"{len(C)} cells.\n")

    # ---------- 2. carve-out: the headline comparison ------------------
    A("\n## 1. Exception / carve-out language — the ContractNLI hypothesis\n")
    A("The ContractNLI paper names negation-by-exception as the dominant failure")
    A(f"of pretrained models. Subset = gold span, or +/-{CTX} chars around it,")
    A(f"containing any of: {', '.join('`'+c+'`' for c in CARVE)}.\n")
    A("**The comparison is the finding**: error rate on the carve-out subset vs the")
    A("non-carve-out subset. Both are restricted to rows that HAVE gold evidence, so")
    A("they are like-for-like; the all-rows figure is shown only for context and is")
    A("not comparable (it includes NotMentioned rows, which have no gold span).\n")
    A("`STRICT` drops `subject to`, which appears in most commercial contracts")
    A("regardless of whether anything is being carved out, and so dilutes the test.\n")
    A("| model | cond | all rows | gold-bearing | carve-out | non-carve | lift | STRICT carve | strict lift |")
    A("|---|---|---|---|---|---|---|---|---|")
    carve_rows = {}
    for (m, cond), rows in sorted(C.items()):
        cw = ncw = cn = ncn = scw = scn = gw = gn = 0
        sub = []
        for r in rows:
            gold = r["gold_evidence"]
            if not gold:
                continue
            t = text_of.get(r["doc_id"], "")
            wrong = (r["parsed_verdict"] != r["gold_verdict"])
            gn += 1; gw += wrong
            if has_carveout(t, gold):
                cn += 1; cw += wrong
                if wrong:
                    sub.append(r)
            else:
                ncn += 1; ncw += wrong
            if has_carveout(t, gold, strict=True):
                scn += 1; scw += wrong
        allw = sum(1 for r in rows if r["parsed_verdict"] != r["gold_verdict"])
        ce = cw / cn if cn else 0; ne = ncw / ncn if ncn else 0
        se = scw / scn if scn else 0
        carve_rows[(m, cond)] = sub
        A(f"| {m.split('/')[-1]} | {cond} | {allw/len(rows):.1%} | {gw/gn:.1%} | "
          f"**{ce:.1%}** (n={cn}) | {ne:.1%} (n={ncn}) | {ce/ne if ne else 0:.2f}x | "
          f"**{se:.1%}** (n={scn}) | {se/ne if ne else 0:.2f}x |")

    # ---------- 5. FN overlap distribution ----------------------------
    A("\n## 2. Evidence false-negative overlap — near-miss or wrong text?\n")
    A("For every evidence FN, token-set overlap of prediction against gold")
    A("(|pred ∩ gold| / |gold|). Settles whether models nearly find the right")
    A("text or retrieve something else entirely.\n")
    A("| model | cond | n FN | p10 | p50 | p90 | % >0.9 | % >0.5 | % =0 (empty pred) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for (m, cond), rows in sorted(C.items()):
        ov = []
        empty = 0
        for r in rows:
            if not r["gold_evidence"]:
                continue
            case, _ = evidence_case(r["gold_evidence"], r["parsed_evidence"])
            if case != "FN":
                continue
            if not r["parsed_evidence"]:
                empty += 1
            o = tok_overlap(r["parsed_evidence"], r["gold_evidence"])
            if o is not None:
                ov.append(o)
        if not ov:
            continue
        ov.sort()
        q = lambda p: ov[min(int(p * len(ov)), len(ov) - 1)]
        A(f"| {m.split('/')[-1]} | {cond} | {len(ov)} | {q(.10):.2f} | {q(.50):.2f} | "
          f"{q(.90):.2f} | {sum(1 for x in ov if x > .9)/len(ov):.0%} | "
          f"{sum(1 for x in ov if x > .5)/len(ov):.0%} | {empty/len(ov):.0%} |")

    # ---------- case dumps --------------------------------------------
    def dump(title, blurb, picker, per_model=args.per_cat, show_ev=False):
        A(f"\n## {title}\n"); A(blurb + "\n")
        for (m, cond), rows in sorted(C.items()):
            hits = [r for r in rows if picker(r)]
            if not hits:
                continue
            A(f"\n### {m.split('/')[-1]} ({cond}) — {len(hits)} cases, showing "
              f"{min(per_model, len(hits))}\n")
            for r in rng.sample(hits, min(per_model, len(hits))):
                A(f"**doc {r['doc_id']} · {r['hypothesis_id']}** — "
                  f"gold `{r['gold_verdict']}`, predicted `{r['parsed_verdict']}`  ")
                A(f"*Q:* {short(question_of.get(r['hypothesis_id'],''), 200)}  ")
                if r["gold_evidence"]:
                    A(f"*Gold evidence:* {short(r['gold_evidence'][0])}  ")
                if show_ev:
                    A(f"*Predicted evidence:* "
                      f"{short(r['parsed_evidence'][0]) if r['parsed_evidence'] else '(none)'}  ")
                A(f"*Model said:* {short(r['raw_output'], 260)}\n")

    # which policies generate the false alarms?
    A("\n## 3. Which policies generate the false conflicts?\n")
    A("Contradiction false positives concentrated by hypothesis, summed across")
    A("all cells. If a handful of policies drive the noise, the fix is targeted.\n")
    fp = Counter(); seen = Counter()
    for (m, cond), rows in C.items():
        for r in rows:
            seen[r["hypothesis_id"]] += 1
            if r["parsed_verdict"] == "Contradiction" and r["gold_verdict"] != "Contradiction":
                fp[r["hypothesis_id"]] += 1
    tot = sum(fp.values())
    A(f"Total Contradiction false positives across all cells: **{tot}**\n")
    A("| hypothesis | false flags | % of all FPs | rate per cell-row | policy |")
    A("|---|---|---|---|---|")
    cum = 0
    for h, n in fp.most_common(8):
        cum += n
        A(f"| {h} | {n} | {n/tot:.0%} | {n/seen[h]:.0%} | "
          f"{short(question_of.get(h,''), 90)} |")
    A(f"\nTop 8 policies account for **{cum/tot:.0%}** of all false conflicts.\n")

    dump("4. Contradiction false positives (the commercial bottleneck)",
         "Gold is Entailment or NotMentioned; the model flagged a conflict. "
         "Every one of these is a clause escalated to a lawyer for nothing.",
         lambda r: r["parsed_verdict"] == "Contradiction"
                   and r["gold_verdict"] != "Contradiction", show_ev=True)

    dump("5. Failed abstention",
         "Gold is NotMentioned; the model forced a verdict on a clause that does "
         "not address the policy at all.",
         lambda r: r["gold_verdict"] == "NotMentioned"
                   and r["parsed_verdict"] in ("Entailment", "Contradiction"),
         show_ev=True)

    dump("6. Verdict right, evidence wrong (the faithfulness failure)",
         "The verdict is correct so the answer looks right, but the cited clause "
         "is not the one the annotator marked. Most dangerous failure in a legal "
         "product: the citation is what a reviewer trusts.",
         lambda r: r["parsed_verdict"] == r["gold_verdict"]
                   and r["gold_evidence"]
                   and evidence_case(r["gold_evidence"], r["parsed_evidence"])[0] == "FN",
         show_ev=True)

    A("\n## 7. Carve-out failures — worked examples\n")
    A("Wrong-verdict cases where the gold clause sits in exception language.\n")
    for (m, cond), sub in sorted(carve_rows.items()):
        if not sub:
            continue
        A(f"\n### {m.split('/')[-1]} ({cond}) — {len(sub)} cases, showing "
          f"{min(8, len(sub))}\n")
        for r in rng.sample(sub, min(8, len(sub))):
            g = r["gold_evidence"][0]
            kw = sorted({k.lower() for k in CARVE_RE.findall(
                short(text_of.get(r["doc_id"], ""), 10**9))} & set(CARVE))
            A(f"**doc {r['doc_id']} · {r['hypothesis_id']}** — gold "
              f"`{r['gold_verdict']}`, predicted `{r['parsed_verdict']}`  ")
            A(f"*Gold evidence:* {short(g)}  ")
            A(f"*Model said:* {short(r['raw_output'], 240)}\n")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"-> {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
