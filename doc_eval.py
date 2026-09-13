#!/usr/bin/env python3
"""
Run one (model, condition) cell of the document-level screen.

  python doc_eval.py --model Qwen/Qwen3-8B --condition nothink --split dev

Conditions:
  nothink - zero-shot, enable_thinking=False   (primary, all models)
  think   - zero-shot, enable_thinking=True    (Qwen3 only)

NO few-shot: at document level each exemplar carries a full contract, so three
exemplars plus the test contract exceed context. ContractEval is zero-shot for
the same reason. Deliberate, not an omission.
"""
import os, json, time, argparse, socket
from pathlib import Path

from doc_harness import build_messages, parse_output

COND = {"nothink": False, "think": True}


def ids(enc):
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def detect_think_kwarg(tok, msgs):
    """Probe BOTH directions: a template whose default is already non-thinking
    renders identically for undefined and False, so a False-only probe misses
    it (Gemma 4). Returns (name, default_is_thinking)."""
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
            return name, (base == on)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=list(COND))
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--data-dir", default="doc_sft")
    ap.add_argument("--out-dir", default=os.environ.get("RESULTS_DIR", "doc_results"))
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 -> 512 for nothink, 2048 for think")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    mnt = args.max_new_tokens or (2048 if args.condition == "think" else 512)
    fn = {"dev": "valid", "test": "test"}[args.split]
    rows = [json.loads(l) for l in
            open(Path(args.data_dir) / f"{fn}.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.model)
    probe = build_messages("x", "y")
    kw, default_thinking = detect_think_kwarg(tok, probe)
    want = COND[args.condition]
    if kw is None and want:
        raise SystemExit(f"{args.model} has no thinking toggle; condition 'think' is N/A")
    extra = {kw: want} if kw else {}

    prompts = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        prompts.append(ids(tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, **extra)))

    plen = sorted(len(p) for p in prompts)
    budget = args.max_model_len - mnt
    over = [(r["doc_id"], r["hypothesis_id"], len(p))
            for r, p in zip(rows, prompts) if len(p) > budget]
    print(f"model        : {args.model}")
    print(f"condition    : {args.condition}  (kwarg={kw}, default_thinking={default_thinking})")
    print(f"split        : {args.split}   n={len(rows)}   max_new_tokens={mnt}")
    print(f"prompt tokens: p50={plen[len(plen)//2]} p99={plen[int(.99*len(plen))]} max={plen[-1]}")
    if over:
        # Never truncate an eval input - report instead.
        print(f"WARNING: {len(over)} inputs exceed max_model_len-max_new_tokens ({budget}):")
        for d, h, n in over[:10]:
            print(f"    doc {d} {h}: {n} tokens")
    print(flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=mnt)

    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    elapsed = time.time() - t0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "__")
    stem = f"{slug}__{args.condition}__{args.split}" + (f"__{args.tag}" if args.tag else "")
    raw_path = Path(args.out_dir) / f"raw__{stem}.jsonl"

    ntok, ntrunc = [], 0
    with open(raw_path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            p = parse_output(g.text)
            ntok.append(len(g.token_ids))
            ntrunc += (g.finish_reason == "length")
            f.write(json.dumps({
                "model": args.model, "condition": args.condition, "split": args.split,
                "doc_id": r["doc_id"], "hypothesis_id": r["hypothesis_id"],
                "file_name": r.get("file_name"),
                "raw_output": g.text,
                "parsed_verdict": p["verdict"], "parsed_evidence": p["evidence"],
                "gold_verdict": r["verdict"], "gold_evidence": r["evidence"],
                "json_state": p["json_state"], "schema_valid": p["schema_valid"],
                "parse_mode": p["parse_mode"],
                "n_output_tokens": len(g.token_ids),
                "finish_reason": g.finish_reason,
                "n_prompt_tokens": len(o.prompt_token_ids or []),
            }, ensure_ascii=False) + "\n")

    ntok.sort()
    meta = {"model": args.model, "condition": args.condition, "split": args.split,
            "n": len(rows), "max_new_tokens": mnt, "think_kwarg": kw,
            "default_is_thinking": default_thinking,
            "mean_output_tokens": round(sum(ntok)/len(ntok), 2),
            "p99_output_tokens": ntok[int(.99*len(ntok))], "max_output_tokens": ntok[-1],
            "truncated": ntrunc,
            "wall_seconds": round(elapsed, 1),
            "sec_per_example": round(elapsed/len(rows), 3),
            "prompt_p50": plen[len(plen)//2], "prompt_max": plen[-1],
            "inputs_over_budget": len(over),
            "host": socket.gethostname()}
    (Path(args.out_dir) / f"meta__{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {raw_path}")
    print(f"  {elapsed:.0f}s  ({meta['sec_per_example']}s/ex)  mean_out={meta['mean_output_tokens']} "
          f"truncated={ntrunc}")


if __name__ == "__main__":
    main()
