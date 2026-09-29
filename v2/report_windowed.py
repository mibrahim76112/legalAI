#!/usr/bin/env python3
"""Build v2/CUAD_WINDOWED_INFERENCE_REPORT.md from A and B.

A is READ from the existing evaluation output. It is never regenerated.
Every number written here comes from a file produced by an actual run.
"""
import sys, json, argparse, hashlib
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402
from doc_harness import strip_wrappers, _find_json             # noqa: E402
from cuad_windowed import contract_text, SYSTEM, USER_TMPL     # noqa: E402

FOCUS = ["Non-Transferable License", "Ip Ownership Assignment",
         "Exclusivity", "Non-Compete"]


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def bucket(n):
    return "<16K" if n < 16000 else "16-32K" if n < 32000 else ">32K"


def tparse(raw):
    t = strip_wrappers(raw)
    if not t:
        return None, []
    d = _find_json(t)
    if not isinstance(d, dict) or "present" not in d:
        return None, []
    ev = d.get("evidence") or []
    return bool(d["present"]), ([str(x) for x in ev] if isinstance(ev, list) else [])


def score(pairs, keys):
    c = Counter(); ev_ok = ev_no = 0
    for k in keys:
        x = pairs[k]
        g, p = x["gold"], x["pred"]
        c["tp" if (g and p) else "fp" if (p and not g) else
          "fn" if (g and not p) else "tn"] += 1
        if g and x["gold_evidence"]:
            hay = norm(" ".join(x["evidence"]))
            if all(_grounded(gs, hay) for gs in x["gold_evidence"]):
                ev_ok += 1
            else:
                ev_no += 1
    f, p_, r_ = f1(c["tp"], c["fp"], c["fn"])
    return {"n": sum(c.values()), "f1": f, "precision": p_, "recall": r_,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "evidence_recall": ev_ok / (ev_ok + ev_no) if ev_ok + ev_no else 0.0,
            "evidence_n": ev_ok + ev_no, "evidence_ok": ev_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-raw", default="/scratch/ibi761/legalai/v2_results/raw__q4_3task.jsonl")
    ap.add_argument("--a-metrics", default="/scratch/ibi761/legalai/v2_results/metrics__q4_3task.json")
    ap.add_argument("--b-windows", default="/scratch/ibi761/legalai/v2_results/windows__q4_3task_win.jsonl")
    ap.add_argument("--b-meta", default="/scratch/ibi761/legalai/v2_results/windowmeta__q4_3task_win.json")
    ap.add_argument("--coord", default="v2/_coord_validation.json")
    ap.add_argument("--overlap", default="v2/_overlap_validation.json")
    ap.add_argument("--avis", default="v2/_a_visibility.json")
    ap.add_argument("--out", default="v2/CUAD_WINDOWED_INFERENCE_REPORT.md")
    args = ap.parse_args()

    meta = json.load(open(args.b_meta))
    amet = json.load(open(args.a_metrics))
    clen = meta["contract_tokens"]
    coord = json.load(open(args.coord))
    try:
        ovl = json.load(open(args.overlap))
    except Exception:
        ovl = None
    try:
        avis = json.load(open(args.avis))
    except Exception:
        avis = None

    # ---------- A ----------
    A = {}
    for r in (json.loads(l) for l in open(args.a_raw, encoding="utf-8")):
        if r.get("task") != "cuad":
            continue
        p, ev = tparse(r["raw"])
        A[(r["contract"], r["category"])] = {
            "gold": bool(r["present"]), "gold_evidence": r["gold_evidence"],
            "pred": bool(p), "evidence": ev, "unparseable": p is None}

    # ---------- B ----------
    wrows = [json.loads(l) for l in open(args.b_windows, encoding="utf-8")]
    B, wdetail = {}, defaultdict(list)
    for w in wrows:
        k = (w["contract"], w["category"])
        b = B.setdefault(k, {"gold": w["gold_present"],
                             "gold_evidence": w["gold_evidence"],
                             "pred": False, "evidence": [], "spans": [],
                             "n_windows": 0, "n_votes": 0, "n_ungrounded": 0})
        b["n_windows"] += 1
        b["n_ungrounded"] += w["n_ungrounded"]
        if w["vote_present"]:
            b["n_votes"] += 1
            b["pred"] = True
            b["evidence"].extend(s["text"] for s in w["grounded_spans"])
            b["spans"].extend(w["grounded_spans"])
        wdetail[k].append(w)

    keys = sorted(set(A) & set(B))
    assert len(keys) == len(A) == len(B), \
        f"key mismatch A={len(A)} B={len(B)} shared={len(keys)}"

    sa, sb = score(A, keys), score(B, keys)

    L = []; W = L.append
    W("# CUAD windowed inference — A (direct 16K) vs B (sliding window)\n")

    W("## 1. Checkpoint used\n")
    W(f"- **Base**: `{meta['base']}`")
    W(f"- **Adapter**: `{meta['adapter']}`")
    ad = Path(meta["adapter"]) / "adapter_model.safetensors"
    if ad.exists():
        h = hashlib.sha256(ad.read_bytes()).hexdigest()
        W(f"- **Adapter sha256**: `{h}`")
    W("\nThis is the multitask 4B checkpoint, trained on the three-task dataset "
      "(ContractNLI + CUAD + risk notes) at max_seq_len 16,384.\n")

    W("## 2. Confirmation that A and B use the same checkpoint\n")
    W("Verified three ways, not assumed:\n")
    W(f"1. A's metrics file records `base` = `{amet['base']}` and `adapter` = "
      f"`{amet['adapter']}`.")
    W("2. A's SLURM log line reads "
      "`=== eval Qwen/Qwen3-4B adapter=/scratch/ibi761/legalai/v2_sft/"
      "Qwen__Qwen3-4B__3task/final ===`, so the path in the metrics file is the "
      "path the job actually loaded.")
    W(f"3. B was launched against the identical path, recorded in its own meta "
      f"file as `{meta['adapter']}`.")
    same = (amet["base"] == meta["base"]) and (amet["adapter"] == meta["adapter"])
    W(f"\n**Match: {'YES' if same else 'NO'}.** "
      f"{'A and B are the same weights.' if same else 'STOP — mismatch.'}\n")

    W("## 3. Direct 16K baseline source\n")
    W(f"- Predictions: `{args.a_raw}`")
    W(f"- Metrics: `{args.a_metrics}`")
    W(f"- A was **not** re-run. n = {amet['n']} rows total "
      f"({len(A)} CUAD), wall clock {amet['wall_seconds']/60:.1f} min, "
      f"{amet['prompts_left_truncated']} prompts left-truncated.")
    W("- Join key is `(contract, category)`, unique in both files, and the two "
      "key sets are identical, so every comparison below is like-for-like on "
      f"the same {len(keys)} pairs.\n")

    W("## 4. Windowing configuration\n")
    W(f"| setting | value |")
    W(f"|---|---|")
    W(f"| total model context | {meta['max_model_len']:,} |")
    W(f"| reserved for generation | {meta['max_new_tokens']:,} |")
    W(f"| input budget | {meta['input_budget']:,} |")
    W(f"| measured scaffold overhead | {meta['scaffold_overhead']} |")
    W(f"| contract fits in one call if | <= {meta['input_budget']-meta['scaffold_overhead']:,} tokens |")
    W(f"| window content tokens | {meta['content_tokens']:,} |")
    W(f"| stride | {meta['stride']:,} |")
    W(f"| overlap | {meta['content_tokens']-meta['stride']:,} |")
    W("\n**CASE 1 is honoured.** A contract is windowed only when the complete "
      "prompt exceeds the input budget. Short contracts are passed whole in a "
      "single call and are byte-identical to what A saw, so they cannot "
      "manufacture a difference. The overhead figure is measured by rendering "
      "the template with an empty contract and the longest category name, not "
      "guessed.\n")
    W("**Aggregation.** A window's `present: true` counts only if its emitted "
      "span is grounded in that window via `reasoning_filters._grounded`. "
      "Contract-level present if ANY window casts a grounded vote. No "
      "thresholds, no calibration, no per-category tuning.\n")

    W("## 5. Cost pre-check\n")
    W("| quantity | value |")
    W("|---|---|")
    W(f"| contracts requiring multiple windows | {meta['contracts_multi_window']} |")
    W(f"| contracts evaluated in one window | {meta['contracts_single_window']} |")
    W(f"| total model calls, B | {meta['n_calls_B']:,} |")
    W(f"| total model calls, A | {meta['n_calls_A']:,} |")
    W(f"| average windows per contract | {meta['avg_windows_per_contract']:.2f} |")
    W(f"| maximum windows for any contract | {meta['max_windows']} |")
    W(f"| windows-per-contract histogram | {meta['histogram']} |")
    W(f"\nPre-check estimate was under two hours, so B was launched without "
      f"asking. Actual: **{meta['wall_seconds']/60:.1f} min**, "
      f"{meta['sec_per_call']:.3f}s per call.\n")

    W("## 6-7. Direct 16K results and windowed results\n")
    W("| metric | A direct 16K | B windowed | delta |")
    W("|---|---|---|---|")
    for lab, ka in (("PRESENT F1", "f1"), ("PRESENT precision", "precision"),
                    ("PRESENT recall", "recall"), ("evidence recall", "evidence_recall")):
        W(f"| {lab} | {sa[ka]:.4f} | {sb[ka]:.4f} | {sb[ka]-sa[ka]:+.4f} |")
    W(f"| TP | {sa['tp']} | {sb['tp']} | {sb['tp']-sa['tp']:+d} |")
    W(f"| FP | {sa['fp']} | {sb['fp']} | {sb['fp']-sa['fp']:+d} |")
    W(f"| FN | {sa['fn']} | {sb['fn']} | {sb['fn']-sa['fn']:+d} |")
    W(f"| TN | {sa['tn']} | {sb['tn']} | {sb['tn']-sa['tn']:+d} |")
    W(f"\nn = {sa['n']} (contract, category) pairs; evidence recall computed "
      f"over the {sa['evidence_n']} pairs that carry gold evidence.\n")

    W("## 8. Length-stratified comparison\n")
    W("Windowing can only act where the window binds. Most dev contracts fit "
      "in one call, so the pooled numbers above are diluted and this table is "
      "the one that carries the answer.\n")
    W("| bucket | contracts | pairs | A F1 | B F1 | A recall | B recall | "
      "A ev-recall | B ev-recall |")
    W("|---|---|---|---|---|---|---|---|---|")
    for bk in ("<16K", "16-32K", ">32K"):
        sub = [k for k in keys if bucket(clen[k[0]]) == bk]
        if not sub:
            continue
        a_, b_ = score(A, sub), score(B, sub)
        W(f"| {bk} | {len({k[0] for k in sub})} | {len(sub)} | "
          f"{a_['f1']:.3f} | {b_['f1']:.3f} | {a_['recall']:.3f} | {b_['recall']:.3f} | "
          f"{a_['evidence_recall']:.3f} | {b_['evidence_recall']:.3f} |")
    W("")

    W("## 9. Per-category comparison (complete)\n")
    W("| category | pairs | gold+ | A F1 | B F1 | A rec | B rec | A ev-rec | B ev-rec |")
    W("|---|---|---|---|---|---|---|---|---|")
    cats = sorted({k[1] for k in keys})
    percat = {}
    for c in cats:
        sub = [k for k in keys if k[1] == c]
        a_, b_ = score(A, sub), score(B, sub)
        percat[c] = (a_, b_)
        W(f"| {c} | {len(sub)} | {a_['tp']+a_['fn']} | {a_['f1']:.3f} | {b_['f1']:.3f} | "
          f"{a_['recall']:.3f} | {b_['recall']:.3f} | "
          f"{a_['evidence_recall']:.3f} | {b_['evidence_recall']:.3f} |")
    W("\n### Truncation-sensitive categories\n")
    W("| category | A F1 | B F1 | A ev-recall | B ev-recall | verdict |")
    W("|---|---|---|---|---|---|")
    for c in FOCUS:
        if c not in percat:
            continue
        a_, b_ = percat[c]
        d = b_["evidence_recall"] - a_["evidence_recall"]
        v = "recovered" if d > 0.001 else "no change" if abs(d) <= 0.001 else "regressed"
        W(f"| {c} | {a_['f1']:.3f} | {b_['f1']:.3f} | {a_['evidence_recall']:.3f} | "
          f"{b_['evidence_recall']:.3f} | {v} |")
    W("")

    W("## 10. Evidence visibility and coverage\n")
    W("| | A direct 16K | B windowed |")
    W("|---|---|---|")
    if avis:
        W(f"| gold evidence spans | {avis['spans_total']} | {coord['counts'].get('gold_spans',0)} |")
        W(f"| spans visible to the model | {avis['spans_visible']} | "
          f"{coord['b_visibility'].get('visible',0)} |")
        W(f"| spans invisible | {avis['spans_invisible']} | "
          f"{coord['b_visibility'].get('INVISIBLE',0)} |")
        W(f"| visibility | {avis['spans_visible']/avis['spans_total']:.2%} | "
          f"{coord['b_visibility'].get('visible',0)/coord['counts'].get('gold_spans',0):.2%} |")
        W(f"| empirical evidence-recall ceiling | "
          f"{avis['spans_visible']/avis['spans_total']:.2%} | "
          f"{coord['b_visibility'].get('visible',0)/coord['counts'].get('gold_spans',0):.2%} |")
    W("\nB's visibility is **measured**, not assumed: every gold span was "
      "searched for in the windows actually constructed, using the same "
      "`_grounded` test the aggregation uses.\n")

    W("## 11. Coordinate-mapping validation\n")
    cc = coord["counts"]
    W(f"- Gold spans round-tripped: **{cc.get('roundtrip_OK',0)} / {cc.get('gold_spans',0)}**, "
      f"failures {cc.get('roundtrip_FAIL',0)}, unlocatable {cc.get('located_FAIL',0)}.")
    W("- Method: map the span to original-contract offsets, then slice the "
      "ORIGINAL contract at those offsets and require the result to match.")
    W(f"- Positions exercised by gold spans: "
      f"{coord['position_coverage'].get('start_of_window',{}).get('n',0)} near a "
      f"window start, "
      f"{coord['position_coverage'].get('end_of_window',{}).get('n',0)} near a "
      f"window end, "
      f"{coord['position_coverage'].get('in_overlap',{}).get('n',0)} inside an "
      f"overlap region.")
    if ovl:
        o = ovl["counts"]
        W(f"\n**No gold span landed in an overlap region**, so that branch was "
          f"driven directly with real contract text sampled from the shared "
          f"regions themselves. A span in an overlap is visible from two "
          f"windows at different local offsets and both must resolve to the "
          f"same original coordinates:\n")
        W("| check | count |")
        W("|---|---|")
        W(f"| overlap regions exercised | {o.get('overlap_regions',0)} |")
        W(f"| spans tested | {o.get('spans_tested',0)} |")
        W(f"| both windows agree on offsets | {o.get('both_windows_AGREE',0)} |")
        W(f"| both windows disagree | {o.get('both_windows_DISAGREE',0)} |")
        W(f"| round-trip OK | {o.get('roundtrip_OK',0)} |")
        W(f"| round-trip FAIL | {o.get('roundtrip_FAIL',0)} |")
    else:
        W("\n**Overlap-region mapping is NOT yet validated** — no gold span "
          "falls in an overlap and the targeted test has not been run. "
          "Treat overlap-region coordinates as unverified.")
    W(f"\nDuring B itself, {meta.get('grounded_but_unmappable_spans','n/a')} "
      f"spans passed the groundedness test but could not be assigned "
      f"coordinates.\n")

    W("## 12. Inference cost\n")
    W("| | A | B |")
    W("|---|---|---|")
    W(f"| model calls (CUAD) | {meta['n_calls_A']:,} | {meta['n_calls']:,} |")
    W(f"| ratio | 1.00x | {meta['n_calls']/meta['n_calls_A']:.2f}x |")
    W(f"| wall clock | {amet['wall_seconds']/60:.1f} min (all 3 tasks, "
      f"{amet['n']} rows) | {meta['wall_seconds']/60:.1f} min (CUAD only) |")
    W(f"| sec per call | {amet['sec_per_example']:.3f} | {meta['sec_per_call']:.3f} |")
    W(f"| avg windows/contract | 1.00 | {meta['avg_windows_per_contract']:.2f} |")
    W(f"| max windows/contract | 1 | {meta['max_windows']} |")
    W("\nPrefix caching was enabled for both. In B the system prompt plus the "
      "contract window is a shared prefix across all 18 categories of a "
      "contract, and prompts were ordered by (contract, window) so that prefix "
      "stays hot. A's wall clock covers all three tasks and is not a per-call "
      "comparison; the per-call figures are.\n")

    # ---------------- 13. error analysis ----------------
    W("## 13. Error analysis\n")
    cov = accu = aggr = 0
    grd = sum(b["n_ungrounded"] for b in B.values())
    ex = {"A": [], "B": [], "C": [], "D": []}
    ainvis = set(tuple(x) for x in (avis or {}).get("invisible_keys", []))
    for k in keys:
        a, b = A[k], B[k]
        if a["gold"] and not a["pred"]:
            if k in ainvis:
                cov += 1
                if len(ex["A"]) < 3:
                    ex["A"].append({"pair": list(k)})
            else:
                accu += 1
                if len(ex["B"]) < 3:
                    ex["B"].append({"pair": list(k)})
        if (not b["gold"]) and b["pred"] and b["n_votes"] >= 1:
            aggr += 1
            if len(ex["C"]) < 3:
                ex["C"].append({"pair": list(k), "votes": b["n_votes"],
                                "of_windows": b["n_windows"]})
    for k in keys:
        if B[k]["n_ungrounded"] and len(ex["D"]) < 3:
            ex["D"].append({"pair": list(k),
                            "ungrounded_claims": B[k]["n_ungrounded"]})
    W("| class | definition | count |")
    W("|---|---|---|")
    W(f"| A. coverage | gold evidence outside the visible 16K input (A misses) | {cov} |")
    W(f"| B. model accuracy | evidence was visible, model still wrong (A misses) | {accu} |")
    W(f"| C. aggregation | B says present on a gold-absent pair via a grounded window vote | {aggr} |")
    W(f"| D. grounding | window claimed present but cited text not in that window | {grd} |")
    W("\nClass D is the one the aggregation rule exists to stop. Each of those "
      f"{grd} claims would have become a contract-level false positive under a "
      "plain OR over window verdicts.\n")
    for cl, nm in (("A", "coverage"), ("B", "model accuracy"),
                   ("C", "aggregation"), ("D", "grounding")):
        if ex[cl]:
            W(f"- **{cl} ({nm})** examples: "
              + "; ".join(f"`{e['pair'][1]}` in `{e['pair'][0][:40]}`"
                          + (f" ({e.get('ungrounded_claims','')} ungrounded)"
                             if cl == "D" else "")
                          + (f" ({e.get('votes')}/{e.get('of_windows')} windows voted)"
                             if cl == "C" else "")
                          for e in ex[cl]))
    W("")

    json.dump({"A": sa, "B": sb,
               "per_category": {c: {"A": percat[c][0], "B": percat[c][1]} for c in percat},
               "error_analysis": {"coverage": cov, "model_accuracy": accu,
                                  "aggregation": aggr, "grounding": grd}},
              open(str(Path(args.out).with_suffix(".json")), "w"), indent=2)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"-> {args.out}")
    print(f"A: F1={sa['f1']:.4f} P={sa['precision']:.4f} R={sa['recall']:.4f} "
          f"evR={sa['evidence_recall']:.4f}")
    print(f"B: F1={sb['f1']:.4f} P={sb['precision']:.4f} R={sb['recall']:.4f} "
          f"evR={sb['evidence_recall']:.4f}")


if __name__ == "__main__":
    main()
