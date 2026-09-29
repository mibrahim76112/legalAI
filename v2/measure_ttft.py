#!/usr/bin/env python3
"""Per-request TTFT and decode rate, which the batched evals never captured.

The evaluations ran everything as one big batch, so vLLM's aggregate
throughput was logged but time-to-first-token was not. TTFT is a per-request,
concurrency-1 property: it is dominated by prefill over the contract, which
for this workload is 2k-12k tokens.
"""
import sys, json, time, argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="v2/quant_subset.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--out-dir", default="/scratch/ibi761/legalai/v2_baseline")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(args.base)
    kw = {}
    probe = [{"role": "user", "content": "x"}]
    for nmk in ("enable_thinking", "thinking"):
        try:
            a = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **{nmk: True})
            b = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **{nmk: False})
        except Exception:
            continue
        if a != b:
            kw = {nmk: False}
            break

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][:args.n]
    prompts = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        e = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
        if hasattr(e, "input_ids"):
            e = e["input_ids"]
        if e and isinstance(e[0], list):
            e = e[0]
        e = list(e)
        prompts.append(e[-(args.max_model_len - 1024):] if len(e) > args.max_model_len - 1024 else e)

    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, trust_remote_code=True,
              enable_lora=bool(args.adapter), max_lora_rank=16,
              enable_prefix_caching=False)      # OFF: caching would fake TTFT
    lr = LoRARequest("a", 1, args.adapter) if args.adapter else None

    # warm up so the first measurement is not compile time
    llm.generate([TokensPrompt(prompt_token_ids=prompts[0][:512])],
                 SamplingParams(temperature=0, max_tokens=4), lora_request=lr)

    out = []
    for p in prompts:
        # TTFT: one request, one token
        t0 = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=p)],
                     SamplingParams(temperature=0, max_tokens=1), lora_request=lr)
        ttft = time.perf_counter() - t0
        # decode rate: same request, 128 tokens, subtract the prefill
        t1 = time.perf_counter()
        o = llm.generate([TokensPrompt(prompt_token_ids=p)],
                         SamplingParams(temperature=0, max_tokens=128), lora_request=lr)
        tot = time.perf_counter() - t1
        n_out = len(o[0].outputs[0].token_ids)
        dec = (n_out - 1) / (tot - ttft) if tot > ttft and n_out > 1 else None
        out.append({"prompt_tokens": len(p), "ttft_s": round(ttft, 3),
                    "out_tokens": n_out, "decode_tok_s": round(dec, 1) if dec else None})
        print(f"  prompt {len(p):6d} tok  TTFT {ttft:6.2f}s  decode "
              f"{dec if dec else float('nan'):6.1f} tok/s", flush=True)

    ok = [r for r in out if r["decode_tok_s"]]
    s = sorted(r["ttft_s"] for r in out)
    summ = {"tag": args.tag, "base": args.base, "adapter": args.adapter or None,
            "n": len(out), "prefix_caching": False,
            "ttft_median_s": s[len(s)//2], "ttft_min_s": s[0], "ttft_max_s": s[-1],
            "decode_tok_s_median": sorted(r["decode_tok_s"] for r in ok)[len(ok)//2] if ok else None,
            "prompt_tokens_median": sorted(r["prompt_tokens"] for r in out)[len(out)//2],
            "rows": out}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(Path(args.out_dir) / f"ttft__{args.tag}.json", "w"), indent=2)
    print(f"\nTTFT median {summ['ttft_median_s']}s  decode median "
          f"{summ['decode_tok_s_median']} tok/s  at median prompt "
          f"{summ['prompt_tokens_median']} tokens")


if __name__ == "__main__":
    main()
