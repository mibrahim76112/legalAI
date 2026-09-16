#!/usr/bin/env python3
"""
Arm B, Part 1 — teacher reasoning generation over the TRAIN split.

Pass 1 is BLIND: the teacher never sees the gold verdict. If it then reaches the
gold verdict on its own, the reasoning genuinely derived that conclusion rather
than rationalising a label it was handed. That agreement test is the only
correctness signal available, because ContractNLI ships no rationales at all --
annotations carry `choice` and `spans` and nothing else.

Pass 2 regenerates ONLY the disagreements with the gold verdict supplied, tagged
conditioned=true so they can be ablated. Those rows are epistemically weaker by
construction: the teacher is justifying a conclusion it did not reach.

Rows are emitted sorted by doc_id so all 17 questions for a contract are
contiguous. With vLLM prefix caching that prefills each contract once instead of
17 times: 15.3M prompt tokens -> 1.0M, a 15.1x reduction, and prefill is most of
the compute here.
"""
import os, json, time, argparse
from pathlib import Path
from collections import defaultdict, Counter

SYSTEM = (
    "You are a senior contracts lawyer reviewing a Non-Disclosure Agreement "
    "against a stated policy position. Work only from the contract text supplied. "
    "Respond with JSON only, in the form "
    '{"reasoning": "...", "verdict": "...", "evidence": [...]}.\n\n'
    "The reasoning field must, in this order: (1) name the governing clause by "
    "its section number or heading; (2) quote the operative language verbatim "
    "from the contract, in quotation marks; (3) apply that language to the stated "
    "position and state which verdict follows.\n\n"
    "Quote only text that appears in the contract. Do not paraphrase inside "
    "quotation marks. If no clause addresses the position, say so explicitly and "
    "return an empty evidence list.\n\n"
    "verdict is exactly one of Entailment, Contradiction, NotMentioned. "
    "evidence is the list of verbatim sentence(s) that justify the verdict, or [] "
    "for NotMentioned."
)
CONDITION_LINE = (
    "\n\nThe correct verdict for this position is {gold}. Derive that verdict from "
    "the contract. If the contract genuinely does not support it, say so in the "
    "reasoning rather than inventing support."
)
USER_TMPL = "CONTRACT: {text}\n\nPOSITION: {hypothesis}"


def ids(enc):
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def detect_think_kwarg(tok, msgs):
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for name in ("enable_thinking", "thinking"):
        try:
            on = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, **{name: True})
            off = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, **{name: False})
        except Exception:
            continue
        if on != off:
            return name
    return None


def load_train(path="doc_sft/train.jsonl"):
    rows = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        u = [m["content"] for m in r["messages"] if m["role"] == "user"][0]
        body = u.split("CONTRACT: ", 1)[1]
        text, question = body.rsplit("\n\nQUESTION: ", 1)
        rows.append({"doc_id": r["doc_id"], "file_name": r["file_name"],
                     "hypothesis_id": r["hypothesis_id"], "text": text,
                     "hypothesis": question, "gold_verdict": r["verdict"],
                     "gold_evidence": r["evidence"]})
    # doc-contiguous so the prefix cache is reused across a document's 17 questions
    rows.sort(key=lambda r: (r["doc_id"], r["hypothesis_id"]))
    return rows


def build_prompt(tok, r, think_kw, gold=None):
    sys_txt = SYSTEM + (CONDITION_LINE.format(gold=gold) if gold else "")
    msgs = [{"role": "system", "content": sys_txt},
            {"role": "user", "content": USER_TMPL.format(text=r["text"],
                                                         hypothesis=r["hypothesis"])}]
    kw = {think_kw: False} if think_kw else {}
    return ids(tok.apply_chat_template(msgs, tokenize=True,
                                       add_generation_prompt=True, **kw))


def parse(raw):
    """Extract {reasoning, verdict, evidence}. Tolerant of fences and prose."""
    import re
    t = re.sub(r"```(?:json)?|```", " ", raw or "").strip()
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        depth = start = None
        depth = 0
        for i, ch in enumerate(t):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(t[start:i + 1]); break
                    except Exception:
                        start = None
    if not isinstance(obj, dict):
        return None
    v = obj.get("verdict")
    v = {"entailment": "Entailment", "contradiction": "Contradiction",
         "notmentioned": "NotMentioned"}.get(
        re.sub(r"[\s_-]", "", str(v)).lower()) if v else None
    ev = obj.get("evidence")
    if isinstance(ev, str):
        ev = [ev]
    ev = [e for e in (ev or []) if isinstance(e, str) and e.strip()]
    return {"reasoning": obj.get("reasoning") or "", "verdict": v, "evidence": ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--pass", dest="phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--out-dir", default="reasoning")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    rows = load_train()

    if args.phase == 2:
        dis = {(d["doc_id"], d["hypothesis_id"])
               for d in map(json.loads, open(outd / "disagreements.jsonl", encoding="utf-8"))}
        rows = [r for r in rows if (r["doc_id"], r["hypothesis_id"]) in dis]
        print(f"[pass2] regenerating {len(rows)} disagreement rows, conditioned on gold")
    if args.limit:
        rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.teacher)
    think_kw = detect_think_kwarg(tok, [{"role": "system", "content": "x"},
                                        {"role": "user", "content": "y"}])
    prompts = [build_prompt(tok, r, think_kw,
                            gold=(r["gold_verdict"] if args.phase == 2 else None))
               for r in rows]
    plen = sorted(len(p) for p in prompts)
    print(f"teacher      : {args.teacher}  (think kwarg {think_kw})")
    print(f"rows         : {len(rows)}   docs: {len({r['doc_id'] for r in rows})}")
    print(f"prompt tokens: p50={plen[len(plen)//2]} p99={plen[int(.99*len(plen))]} max={plen[-1]}")
    over = [p for p in plen if p > args.max_model_len - args.max_new_tokens]
    if over:
        print(f"WARNING {len(over)} prompts exceed budget; longest {over[-1]}")

    llm = LLM(model=args.teacher, dtype="bfloat16", tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
              trust_remote_code=True, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    el = time.time() - t0

    raw_path = outd / f"pass{args.phase}_raw.jsonl"
    agree = Counter(); per_class = defaultdict(Counter); disagreements = []
    with open(raw_path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            p = parse(g.text) or {"reasoning": "", "verdict": None, "evidence": []}
            ok = (p["verdict"] == r["gold_verdict"])
            agree[ok] += 1
            per_class[r["gold_verdict"]][ok] += 1
            rec = {**{k: r[k] for k in ("doc_id", "file_name", "hypothesis_id",
                                        "hypothesis", "gold_verdict", "gold_evidence")},
                   "teacher": args.teacher, "pass": args.phase,
                   "conditioned": args.phase == 2,
                   "raw_output": g.text, "reasoning": p["reasoning"],
                   "verdict": p["verdict"], "evidence": p["evidence"],
                   "agrees_with_gold": ok,
                   "n_output_tokens": len(g.token_ids),
                   "finish_reason": g.finish_reason}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.phase == 1 and not ok:
                disagreements.append(rec)

    tot = sum(agree.values())
    print(f"\nwrote {raw_path}  ({el/60:.1f} min, {el/len(rows):.2f}s/row)")
    print(f"agreement with gold: {agree[True]}/{tot} = {agree[True]/tot:.1%}")
    for c in ("Entailment", "Contradiction", "NotMentioned"):
        n = sum(per_class[c].values())
        if n:
            print(f"   {c:14s} {per_class[c][True]}/{n} = {per_class[c][True]/n:.1%}")
    if args.phase == 1:
        dp = outd / "disagreements.jsonl"
        with open(dp, "w", encoding="utf-8") as f:
            for d in disagreements:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"disagreements -> {dp} ({len(disagreements)} rows) for label audit + pass 2")


if __name__ == "__main__":
    main()
