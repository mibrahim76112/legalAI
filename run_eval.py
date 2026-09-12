#!/usr/bin/env python3
"""
Run one (model, condition) cell of the screen with vLLM and write raw outputs.

  python run_eval.py --model Qwen/Qwen3-8B --condition threeshot

Fairness controls held fixed across all four models:
  greedy (temperature 0), BF16, one shared max_new_tokens cap,
  each model's OWN chat template, one shared parser, no per-model prompt tuning.
"""
import os, json, time, argparse, socket
from pathlib import Path

from harness_core import build_messages, parse_verdict, SYSTEM_PROMPTS

CLASSES = ["Entailment", "Contradiction", "NotMentioned"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _ids(enc):
    """transformers>=5 returns a BatchEncoding from tokenize=True, not a list."""
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    # a batch dim sneaks in for some tokenizers
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def detect_think_kwarg(tok, msgs):
    """Find the kwarg that puts this model's template in direct-answer mode.

    Qwen3 defaults its template to a reasoning mode that emits a <think> block
    before the answer; under a shared token cap that truncates before the JSON
    ever appears, scoring a capable model as unparseable. That is a harness
    artefact, not a capability gap, so we ask for direct-answer mode.

    Detection compares rendered text, because templates silently IGNORE unknown
    kwargs rather than raising - so a TypeError probe reports false support.
    Returns the kwarg name, or None if this template has no such switch.
    """
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for name in ("enable_thinking", "thinking"):
        try:
            alt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, **{name: False})
        except Exception:
            continue
        if alt != base:
            return name
    return None


def templated_ids(tok, msgs, think_kwarg=None):
    kw = {think_kwarg: False} if think_kwarg else {}
    return _ids(tok.apply_chat_template(msgs, tokenize=True,
                                        add_generation_prompt=True, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["zeroshot", "threeshot"])
    ap.add_argument("--eval-file", default="data/eval_dev.jsonl")
    ap.add_argument("--fewshot-file", default="data/fewshot.json")
    ap.add_argument("--out-dir", default=os.environ.get("RESULTS_DIR", "results"))
    ap.add_argument("--system", default="spec", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N examples")
    ap.add_argument("--allow-thinking", action="store_true",
                    help="do NOT request direct-answer mode (diagnostic only)")
    args = ap.parse_args()

    rows = load_jsonl(args.eval_file)
    if args.limit:
        rows = rows[:args.limit]
    fewshot = json.load(open(args.fewshot_file, encoding="utf-8")) \
        if args.condition == "threeshot" else None

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)

    probe = build_messages(rows[0]["premise"], rows[0]["hypothesis"], fewshot,
                           system=SYSTEM_PROMPTS[args.system])
    think_kw = None if args.allow_thinking else detect_think_kwarg(tok, probe)

    prompts = [templated_ids(tok, build_messages(r["premise"], r["hypothesis"], fewshot,
                                                 system=SYSTEM_PROMPTS[args.system]),
                             think_kw)
               for r in rows]

    plen = [len(p) for p in prompts]
    print(f"model      : {args.model}")
    print(f"condition  : {args.condition}   system-prompt: {args.system}")
    print(f"examples   : {len(rows)}")
    print(f"direct-answer kwarg: {think_kw or 'not supported by this template'}")
    print(f"prompt tokens: min={min(plen)} med={sorted(plen)[len(plen)//2]} max={max(plen)}",
          flush=True)
    if max(plen) > args.max_model_len:
        raise SystemExit(f"prompt ({max(plen)}) exceeds max_model_len ({args.max_model_len})")

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enforce_eager=False)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    t0 = time.time()
    from vllm import TokensPrompt
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    elapsed = time.time() - t0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "__")
    # system tag in the filename: without it a --system sft run silently
    # overwrites the spec results, which are the ranking of record.
    stem = f"{slug}__{args.condition}__{args.system}"
    raw_path = Path(args.out_dir) / f"raw__{stem}.jsonl"

    n_tok = 0
    with open(raw_path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            verdict, json_valid, how = parse_verdict(g.text)
            n_tok += len(g.token_ids)
            f.write(json.dumps({
                "model": args.model, "condition": args.condition,
                "system_prompt": args.system,
                "example_id": r["example_id"], "chunk_id": r["chunk_id"],
                "hypothesis_id": r["hypothesis_id"],
                "raw_output": g.text,
                "parsed_verdict": verdict, "gold_verdict": r["gold_verdict"],
                "json_valid": json_valid, "parse_mode": how,
                "n_output_tokens": len(g.token_ids),
                "finish_reason": g.finish_reason,
            }, ensure_ascii=False) + "\n")

    meta = {
        "model": args.model, "condition": args.condition, "system_prompt": args.system,
        "n_examples": len(rows), "max_new_tokens": args.max_new_tokens,
        "direct_answer_kwarg": think_kw, "mean_output_tokens": round(n_tok/len(rows), 2),
        "wall_seconds": round(elapsed, 1), "host": socket.gethostname(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "prompt_tokens_median": sorted(plen)[len(plen)//2],
    }
    (Path(args.out_dir) / f"meta__{stem}.json").write_text(
        json.dumps(meta, indent=2))
    print(f"\nwrote {raw_path}  ({elapsed:.0f}s, mean out tokens "
          f"{meta['mean_output_tokens']})")


if __name__ == "__main__":
    main()
