#!/usr/bin/env python3
"""Generate the Phase 1 results tables. No hand-transcribed numbers."""
import json, glob, csv, argparse, itertools
from pathlib import Path
from doc_metrics import cell_metrics, paired_boot, boot_ci, UNPARSED
from doc_harness import CLASSES

SHORT = {"nothink": "A zero-shot", "think": "B thinking"}


def load(fp):
    r = [json.loads(l) for l in open(fp, encoding="utf-8")]
    r.sort(key=lambda x: (x["doc_id"], x["hypothesis_id"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/scratch/ibi761/legalai/doc_results")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="PHASE1_RESULTS.md")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    cells, metas = {}, {}
    for fp in sorted(glob.glob(str(Path(args.dir) / f"raw__*__{args.split}.jsonl"))):
        rows = load(fp)
        if not rows:
            continue
        cells[(rows[0]["model"], rows[0]["condition"])] = rows
        mp = fp.replace("raw__", "meta__").replace(".jsonl", ".json")
        if Path(mp).exists():
            metas[(rows[0]["model"], rows[0]["condition"])] = json.load(open(mp))
    if not cells:
        raise SystemExit(f"no raw__*__{args.split}.jsonl in {args.dir}")

    recs = {}
    for k, rows in cells.items():
        m = cell_metrics(rows)
        gold = [r["gold_verdict"] for r in rows]
        pred = [r["parsed_verdict"] or UNPARSED for r in rows]
        m["ci"] = boot_ci(pred, gold, args.bootstrap)
        m.update({f"meta_{a}": b for a, b in metas.get(k, {}).items()
                  if a in ("wall_seconds", "sec_per_example", "truncated",
                           "max_output_tokens", "inputs_over_budget")})
        recs[k] = m

    L = []; A = L.append
    A(f"# Phase 1 — document-level pre-SFT screen ({args.split} split)\n")
    A("Full NDA as context, one policy question per call, **zero-shot**.")
    A("Task supersedes the clause-level screen; those numbers measured an easier")
    A("task and are not mixed in here.\n")
    A("**No few-shot by design.** At document level each exemplar carries a full")
    A("contract, so three exemplars plus the test contract exceed context.")
    A("ContractEval is zero-shot for the same reason. Deliberate, not an omission.\n")
    n = len(next(iter(cells.values())))
    A(f"n = {n} (document, hypothesis) pairs per cell.\n")

    # ---- primary ----
    A("\n## Ranking — macro-F1, condition A (zero-shot non-thinking)\n")
    A("| model | cond | macro-F1 | 95% CI | acc | Contradiction recall | NotMentioned recall |")
    A("|---|---|---|---|---|---|---|")
    order = sorted(recs, key=lambda k: -recs[k]["macro_f1"])
    for k in order:
        m = recs[k]
        A(f"| {k[0].split('/')[-1]} | {SHORT[k[1]]} | **{m['macro_f1']:.3f}** | "
          f"[{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] | {m['accuracy']:.3f} | "
          f"{m['Contradiction_recall']:.3f} | {m['NotMentioned_recall']:.3f} |")

    A("\n## Verdict — per class\n")
    A("| model | cond | " + " | ".join(f"{c} P/R/F1" for c in CLASSES) + " |")
    A("|---|---|" + "---|" * len(CLASSES))
    for k in order:
        m = recs[k]
        cs = " | ".join(f"{m[f'{c}_precision']:.2f}/{m[f'{c}_recall']:.2f}/{m[f'{c}_f1']:.2f}"
                        for c in CLASSES)
        A(f"| {k[0].split('/')[-1]} | {SHORT[k[1]]} | {cs} |")

    A("\n## Evidence extraction\n")
    A("Strict = every gold span covered (ContractEval-comparable). "
      "Partial = mean fraction of gold spans covered.\n")
    A("| model | cond | F1 strict | F2 strict | partial | Jaccard | false-abstention | TP/FN/FP/TN |")
    A("|---|---|---|---|---|---|---|---|")
    for k in order:
        m = recs[k]
        A(f"| {k[0].split('/')[-1]} | {SHORT[k[1]]} | {m['ev_f1_strict']:.3f} | "
          f"{m['ev_f2_strict']:.3f} | {m['ev_partial_credit']:.3f} | {m['ev_jaccard']:.3f} | "
          f"{m['ev_false_abstention_rate']:.3f} | "
          f"{m['ev_tp']}/{m['ev_fn']}/{m['ev_fp']}/{m['ev_tn']} |")

    A("\n## JOINT — verdict correct AND all gold spans covered\n")
    A("The product metric. `lenient` is the brief's literal definition, under which")
    A("empty gold counts as 'all covered'. `strict` additionally requires the model")
    A("not to invent evidence when gold is empty. Both shown because the gap is the")
    A("hallucinated-citation rate on NotMentioned rows.\n")
    A("| model | cond | joint strict | joint lenient |")
    A("|---|---|---|---|")
    for k in order:
        m = recs[k]
        A(f"| {k[0].split('/')[-1]} | {SHORT[k[1]]} | **{m['joint_strict']:.3f}** | "
          f"{m['joint_lenient']:.3f} |")

    A("\n## Diagnostics (not for ranking)\n")
    A("| model | cond | schema valid | not JSON | JSON wrong schema | parse fail | truncated | mean out | p99 out | s/example |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        m = recs[k]
        A(f"| {k[0].split('/')[-1]} | {SHORT[k[1]]} | {m['fmt_schema_valid']:.1%} | "
          f"{m['fmt_not_json']:.1%} | {m['fmt_json_wrong_schema']:.1%} | "
          f"{m['fmt_parse_fail']:.1%} | {m['fmt_truncated']:.1%} | "
          f"{m['mean_output_tokens']:.0f} | {m['p99_output_tokens']} | "
          f"{m.get('meta_sec_per_example','-')} |")

    # ---- noise ----
    A("\n## Is the ordering real? (paired bootstrap, condition A)\n")
    a_cells = {k[0]: v for k, v in cells.items() if k[1] == "nothink"}
    if len(a_cells) >= 2:
        gold = [r["gold_verdict"] for r in next(iter(a_cells.values()))]
        preds = {m: [r["parsed_verdict"] or UNPARSED for r in rows]
                 for m, rows in a_cells.items()}
        o = sorted(preds, key=lambda m: -recs[(m, "nothink")]["macro_f1"])
        A("| comparison | diff | 95% CI | verdict |")
        A("|---|---|---|---|")
        for a, b in itertools.combinations(o, 2):
            d = paired_boot(preds[a], preds[b], gold, args.bootstrap)
            tag = "**separated**" if d["ci_low"] > 0 else "within noise"
            A(f"| {a.split('/')[-1]} vs {b.split('/')[-1]} | {d['mean_diff']:+.3f} | "
              f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {tag} |")

    Path(args.out).write_text("\n".join(L) + "\n")
    with open(f"data/phase1_{args.split}.csv", "w", newline="") as f:
        flat = [{"model": k[0], "condition": k[1],
                 **{a: b for a, b in v.items() if a != "ci"}} for k, v in recs.items()]
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(f"-> {args.out}  ({len(L)} lines), data/phase1_{args.split}.csv")
    for k in order:
        print(f"  {k[0]:30s} {SHORT[k[1]]:12s} macroF1={recs[k]['macro_f1']:.3f} "
              f"joint={recs[k]['joint_strict']:.3f}")


if __name__ == "__main__":
    main()
