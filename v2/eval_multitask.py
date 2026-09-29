#!/usr/bin/env python3
"""Evaluate a trained adapter on all three tasks, each with its own metric.

There is deliberately NO single combined score. The three tasks are measured in
different units against different baselines; averaging them would produce a
number that moves for reasons nobody can attribute.

Metric policy, fixed before any result was seen:
  Task 1 ContractNLI -- macro-F1 over the three verdicts, plus strict evidence
      F1. Contradiction PRECISION is reported separately: the pre-SFT screen
      established that recall is adequate everywhere (0.64-0.84) and precision
      is the bottleneck (0.28-0.58), so precision is what SFT must move.
  Task 2 CUAD -- F1 on the PRESENT class, per category, macro-averaged.
      Pooled binary accuracy is NEVER reported alone: 34.5% of rows are
      present, so a constant-absent predictor scores 65.5% and looks strong.
      That trivial baseline is printed beside every pooled figure.
  Task 3 risk notes -- imitation fidelity ONLY. There is no human ground truth;
      the reference is the teacher's own note. ROUGE-L against it measures
      agreement with Qwen3-32B, not correctness. Reported alongside format
      conformance and quote groundedness, and labelled as such.
"""
import os, sys, json, argparse, time, re
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                    # noqa: E402
from doc_harness import (parse_output, evidence_case, CLASSES,  # noqa: E402
                         strip_wrappers, _find_json)
from doc_metrics import (rouge_l, token_overlap, boot_ci,      # noqa: E402
                         UNPARSED)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def parse_cuad(raw):
    """-> (present|None, evidence list). None means unparseable.

    Uses the same wrapper-stripping and JSON extraction as Task 1 rather than a
    second private regex, so "unparseable" means the same thing on both tasks.
    """
    text = strip_wrappers(raw)
    if not text:
        return None, []
    d = _find_json(text)
    if not isinstance(d, dict) or "present" not in d:
        return None, []
    ev = d.get("evidence") or []
    if not isinstance(ev, list):
        ev = []
    return bool(d["present"]), [str(x) for x in ev]


_SENT = re.compile(r"[.!?](?:\s|$)")




def soft_evidence(pairs):
    """Token-level companions to the strict all-or-nothing evidence metric.

    Strict scores a near-miss identically to retrieving unrelated text: one
    missed span makes the whole row a false negative. These are computed ONLY
    over gold-bearing rows -- on a gold-empty row the correct output is an
    empty list, and token overlap is undefined there, so including them would
    manufacture a score out of correct abstentions.

    Reported ALONGSIDE strict, never instead of it: strict is what compares to
    ContractEval, and softening the metric after seeing the strict number would
    be moving the goalposts.
    """
    ro, to, n = [], [], 0
    for gold, pred in pairs:
        if not gold:
            continue
        n += 1
        r = rouge_l(gold, pred)
        t = token_overlap(gold, pred)
        ro.append(r["f"] if r else 0.0)
        to.append(t["f"] if t else 0.0)
    if not n:
        return {}
    return {"n_gold_bearing": n,
            "rouge_l_f_mean": sum(ro) / n,
            "token_overlap_f_mean": sum(to) / n,
            "token_overlap_p_mean": sum(
                (token_overlap(g, p) or {"p": 0.0})["p"] for g, p in pairs if g) / n,
            "token_overlap_r_mean": sum(
                (token_overlap(g, p) or {"r": 0.0})["r"] for g, p in pairs if g) / n}


def score(recs, tag, args, el, over=None):
    """All metrics, given records. Separated from generation so an archived
    raw__*.jsonl can be rescored without spending GPU time -- v1 needed exactly
    this when a format metric turned out to be defective after the fact."""
    M = {"tag": tag, "base": args.base, "adapter": args.adapter or None,
         "inference_quantization": args.quantization or "bf16",
         "data": args.data, "n": len(recs),
         "wall_seconds": round(el, 1), "sec_per_example": round(el / len(recs), 3),
         "prompts_left_truncated": over}

    # ---------------- Task 1 : ContractNLI ----------------
    t1 = [r for r in recs if r["task"] == "contractnli"]
    if t1:
        pred, gold, evrows = [], [], []
        jstate, schema_ok = Counter(), 0
        for r in t1:
            d = parse_output(r["raw"])
            jstate[d["json_state"]] += 1
            schema_ok += bool(d["schema_valid"])
            pred.append(d["verdict"] or UNPARSED)
            gold.append(r["verdict"])
            evrows.append({"parsed_evidence": d["evidence"],
                           "gold_evidence": r["gold_evidence"]})
        per = {}
        for c in CLASSES:
            tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
            fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
            fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
            per[c] = dict(zip(("f1", "precision", "recall"), f1(tp, fp, fn)))
        cases = [evidence_case(r["gold_evidence"], r["parsed_evidence"])
                 for r in evrows]          # -> (case, fraction_covered)
        case = Counter(c for c, _ in cases)
        gold_bearing = [f for (c, f), r in zip(cases, evrows) if r["gold_evidence"]]
        partial1 = sum(gold_bearing) / len(gold_bearing) if gold_bearing else 0.0
        ef, ep, er = f1(case["TP"], case["FP"], case["FN"])
        M["task1_contractnli"] = {
            "n": len(t1),
            "accuracy": sum(1 for p, g in zip(pred, gold) if p == g) / len(t1),
            "macro_f1": sum(per[c]["f1"] for c in CLASSES) / len(CLASSES),
            "macro_f1_ci95": boot_ci(pred, gold, 2000),
            "per_class": per,
            "contradiction_precision": per["Contradiction"]["precision"],
            "contradiction_recall": per["Contradiction"]["recall"],
            "unparseable": sum(1 for p in pred if p == UNPARSED),
            "json_state": dict(jstate),
            "schema_valid_rate": schema_ok / len(t1),
            "evidence_strict_f1": ef, "evidence_precision": ep, "evidence_recall": er,
            "evidence_partial_coverage": partial1,
            "evidence_soft": soft_evidence(
                [(r["gold_evidence"], r["parsed_evidence"]) for r in evrows]),
            "evidence_cases": dict(case)}

    # ---------------- Task 2 : CUAD ----------------
    t2 = [r for r in recs if r["task"] == "cuad"]
    if t2:
        bycat = defaultdict(lambda: Counter())
        unp = 0
        evcase = Counter()
        partials2 = []
        soft2 = []
        for r in t2:
            p, ev = parse_cuad(r["raw"])
            if p is None:
                unp += 1
                p = False                    # unparseable counts as a miss
            g = bool(r["present"])
            c = bycat[r["category"]]
            c["tp" if (p and g) else "fp" if (p and not g) else
              "fn" if (not p and g) else "tn"] += 1
            soft2.append((r["gold_evidence"], ev))
            if g:
                c2, frac2 = evidence_case(r["gold_evidence"], ev)
                evcase[c2] += 1
                partials2.append(frac2)
        percat = {}
        for cat, c in bycat.items():
            fsc, pr, rc = f1(c["tp"], c["fp"], c["fn"])
            percat[cat] = {"f1_present": fsc, "precision": pr, "recall": rc,
                           "support_present": c["tp"] + c["fn"], "n": sum(c.values())}
        tot = Counter()
        for c in bycat.values():
            tot.update(c)
        pooled_f, pooled_p, pooled_r = f1(tot["tp"], tot["fp"], tot["fn"])
        n2 = sum(tot.values())
        n_present = tot["tp"] + tot["fn"]
        ef2, ep2, er2 = f1(evcase["TP"], evcase["FP"], evcase["FN"])
        M["task2_cuad"] = {
            "n": n2, "present_rate": n_present / n2,
            "PRIMARY_macro_f1_present": sum(v["f1_present"] for v in percat.values()) / len(percat),
            "pooled_f1_present": pooled_f,
            "pooled_precision": pooled_p, "pooled_recall": pooled_r,
            "pooled_accuracy": (tot["tp"] + tot["tn"]) / n2,
            "TRIVIAL_constant_absent_accuracy": (n2 - n_present) / n2,
            "unparseable": unp,
            "evidence_strict_f1_on_present": ef2,
            "evidence_partial_coverage": (sum(partials2) / len(partials2)
                                          if partials2 else 0.0),
            "evidence_soft": soft_evidence(soft2),
            "evidence_precision": ep2, "evidence_recall": er2,
            "per_category": dict(sorted(percat.items(),
                                        key=lambda kv: -kv[1]["f1_present"]))}

    # ---------------- Task 3 : risk notes ----------------
    t3 = [r for r in recs if r["task"] == "risknote"]
    if t3:
        rs, one_sent, ungrounded, lens = [], 0, 0, []
        for r in t3:
            note = " ".join((r["raw"] or "").strip().split())
            rl = rouge_l([r["gold_target"]], [note])   # -> dict or None
            rs.append(rl["f"] if rl else 0.0)
            ns = len([x for x in _SENT.split(note) if x.strip()])
            one_sent += (ns == 1)
            lens.append(len(note.split()))
            hay = norm(r["gold_target"])
            for q in re.findall(r'"([^"]{8,})"', note):
                if norm(q) not in hay:
                    ungrounded += 1
                    break
        rs_sorted = sorted(rs)
        M["task3_risknote"] = {
            "n": len(t3),
            "NOTE": "imitation fidelity vs the Qwen3-32B teacher, NOT correctness",
            "rouge_l_mean": sum(rs) / len(rs),
            "rouge_l_median": rs_sorted[len(rs_sorted) // 2],
            "exactly_one_sentence_rate": one_sent / len(t3),
            "mean_words": sum(lens) / len(lens),
            "notes_with_ungrounded_quote": ungrounded}
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="", help="LoRA dir; omit for base model")
    ap.add_argument("--data", default="v2/combined/dev.jsonl")
    ap.add_argument("--out-dir", default="/scratch/ibi761/legalai/v2_results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--quantization", default="",
                    help="vLLM on-the-fly weight quantization for INFERENCE: "
                         "'fp8' (H100 native 8-bit) or 'bitsandbytes' (NF4 "
                         "4-bit). Empty = bf16. The ADAPTER is unchanged; only "
                         "the frozen base weights are quantized, which is what "
                         "isolates the post-training-quantization effect.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescore", default="",
                    help="path to an existing raw__*.jsonl; scores it and "
                         "exits without loading a model")
    args = ap.parse_args()

    if args.rescore:
        recs = [json.loads(l) for l in open(args.rescore, encoding="utf-8")]
        tag = args.tag or Path(args.rescore).stem.replace("raw__", "")
        M = score(recs, tag, args, 0.0)
        outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
        mfp = outd / f"metrics__{tag}.json"
        json.dump(M, open(mfp, "w"), indent=2)
        for k in ("task1_contractnli", "task2_cuad", "task3_risknote"):
            if k in M:
                d = {a: b for a, b in M[k].items()
                     if a not in ("per_category", "per_class")}
                print(f"--- {k}\n" + json.dumps(d, indent=2))
        print(f"-> {mfp}")
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[data] {len(rows)} rows: {dict(Counter(r['task'] for r in rows))}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    probe = [{"role": "user", "content": "x"}]
    kw = {}
    for name in ("enable_thinking", "thinking"):
        try:
            on = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **{name: True})
            off = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **{name: False})
        except Exception:
            continue
        if on != off:
            kw = {name: False}
            break
    print(f"[tmpl] {kw}", flush=True)

    prompts, over = [], 0
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
        if hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        if enc and isinstance(enc[0], list):
            enc = enc[0]
        enc = list(enc)
        budget = args.max_model_len - args.max_new_tokens
        if len(enc) > budget:
            over += 1
            enc = enc[-budget:]          # left-truncate, same rule as training
        prompts.append(enc)
    print(f"[data] {over} prompts left-truncated to fit {args.max_model_len}", flush=True)

    qkw = {"quantization": args.quantization} if args.quantization else {}
    print(f"[quant] inference weights: {args.quantization or 'bf16 (none)'}", flush=True)
    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enable_lora=bool(args.adapter), max_lora_rank=16,
              enable_prefix_caching=True, **qkw)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    lr = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                        lora_request=lr)
    el = time.time() - t0

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (Path(args.adapter).parent.name if args.adapter else "base")
    raw_fp = outd / f"raw__{tag}.jsonl"
    recs = []
    with open(raw_fp, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            txt = o.outputs[0].text or ""
            gold = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
            rec = {"task": r["task"], "raw": txt, "gold_target": gold,
                   "gold_evidence": r.get("evidence") or [],
                   "category": r.get("category"), "verdict": r.get("verdict"),
                   "present": r.get("present"), "contract": r.get("contract"),
                   "doc_id": r.get("doc_id"),
                   "n_out_tokens": len(o.outputs[0].token_ids)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)

    M = score(recs, tag, args, el, over)
    mfp = outd / f"metrics__{tag}.json"
    json.dump(M, open(mfp, "w"), indent=2)
    print(json.dumps({k: v for k, v in M.items() if not k.startswith("task")}, indent=2))
    for k in ("task1_contractnli", "task2_cuad", "task3_risknote"):
        if k in M:
            d = {a: b for a, b in M[k].items() if a not in ("per_category", "per_class")}
            print(f"\n--- {k}\n" + json.dumps(d, indent=2))
    print(f"\n-> {mfp}\n-> {raw_fp}")


if __name__ == "__main__":
    main()
