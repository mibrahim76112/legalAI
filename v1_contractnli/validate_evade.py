#!/usr/bin/env python3
"""
EVADE Phase 2 — validate conditioned rationales.

The generator (Qwen3-32B) wrote a rationale FOR the gold label. The validator
asks whether that rationale actually supports the label, and only rows failing
validation are dropped. That removes a small flagged subset rather than the
majority, which is the whole point of EVADE over an agreement filter.

VALIDATOR INDEPENDENCE: Mixtral-8x22B-Instruct (141B total, ~39B active,
apache-2.0). Different model family from BOTH the Qwen3-32B generator and the
Qwen3-4B/8B/14B students, so it is neither judging its own output nor
contaminating the downstream comparison.

Three strictly binary questions, reported SEPARATELY:
  (a) does the rationale reference text that appears in the contract?
  (b) does the rationale's own logic support the assigned label?
  (c) given the contract, is the assigned label defensible at all?

Only (c) is evidence about the DATA. (a) and (b) are rationale-quality checks --
close cousins of the old f1/f2 string filters that did two thirds of the
original pipeline's cutting. If failures are dominated by (a)/(b) we have
rediscovered teacher articulacy under a new name, not label noise.
"""
import os, json, re, time, argparse, socket
from pathlib import Path
from collections import Counter, defaultdict

SYSTEM = (
    "You are a senior contracts lawyer auditing another lawyer's work. You will "
    "be shown a CONTRACT, a POSITION, the LABEL assigned to that position, and "
    "the RATIONALE written to justify that label.\n\n"
    "Answer exactly three questions, each strictly Yes or No, each followed by "
    "one short sentence of justification.\n\n"
    "A: Does the RATIONALE quote or reference language that actually appears in "
    "the CONTRACT?\n"
    "B: Does the RATIONALE's own reasoning support the assigned LABEL?\n"
    "C: Given the CONTRACT, is the assigned LABEL defensible at all?\n\n"
    "Answer in exactly this format and nothing else:\n"
    "A: Yes|No - <one sentence>\n"
    "B: Yes|No - <one sentence>\n"
    "C: Yes|No - <one sentence>"
)
USER = ("CONTRACT:\n{text}\n\nPOSITION: {hyp}\n\nLABEL: {label}\n\n"
        "RATIONALE: {rat}")

_ANS = {k: re.compile(rf"^\s*{k}\s*[::]\s*(yes|no)\b", re.I | re.M) for k in "ABC"}


def parse_answers(out):
    """-> dict with a/b/c booleans (None if unreadable) and the justifications."""
    res = {}
    for k in "ABC":
        m = _ANS[k].search(out or "")
        res[k.lower()] = (m.group(1).lower() == "yes") if m else None
    just = {}
    for k in "ABC":
        m = re.search(rf"^\s*{k}\s*[::]\s*(?:yes|no)\b[\s\-–—:]*(.+)$",
                      out or "", re.I | re.M)
        just[k.lower()] = (m.group(1).strip()[:200] if m else "")
    return res, just


def ids(enc):
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validator", default="mistralai/Mixtral-8x22B-Instruct-v0.1")
    ap.add_argument("--rationales", default="reasoning/pass2_raw_cond.jsonl")
    ap.add_argument("--data", default="doc_sft/train.jsonl")
    ap.add_argument("--out", default="evade/validated.jsonl")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    rats = [json.loads(l) for l in open(args.rationales, encoding="utf-8")]
    contracts = {}
    for l in open(args.data, encoding="utf-8"):
        x = json.loads(l)
        u = [m["content"] for m in x["messages"] if m["role"] == "user"][0]
        contracts[x["doc_id"]] = u.split("CONTRACT: ", 1)[1].rsplit("\n\nQUESTION: ", 1)[0]
    rats.sort(key=lambda r: (r["doc_id"], r["hypothesis_id"]))
    if args.nshards > 1:
        rats = [r for i, r in enumerate(rats) if i % args.nshards == args.shard]
    if args.limit:
        rats = rats[:args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(args.validator)

    prompts = []
    for r in rats:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(
                    text=contracts.get(r["doc_id"], ""), hyp=r["hypothesis"],
                    label=r["gold_verdict"], rat=" ".join(r["reasoning"].split()))}]
        try:
            p = ids(tok.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True))
        except Exception:      # some templates reject a system role
            merged = [{"role": "user",
                       "content": SYSTEM + "\n\n" + msgs[1]["content"]}]
            p = ids(tok.apply_chat_template(merged, tokenize=True,
                                            add_generation_prompt=True))
        prompts.append(p)
    plen = sorted(len(p) for p in prompts)
    budget = args.max_model_len - args.max_new_tokens
    keep = [i for i, p in enumerate(prompts) if len(p) <= budget]
    if len(keep) < len(prompts):
        print(f"  dropping {len(prompts)-len(keep)} rows over context budget "
              f"(longest {plen[-1]})")
    rats = [rats[i] for i in keep]; prompts = [prompts[i] for i in keep]
    print(f"validator : {args.validator}  tp={args.tp}")
    print(f"rows      : {len(rats)}  shard {args.shard}/{args.nshards}")
    print(f"prompt tok: p50={plen[len(plen)//2]} p99={plen[int(.99*len(plen))]} max={plen[-1]}",
          flush=True)

    llm = LLM(model=args.validator, dtype="bfloat16", tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
              trust_remote_code=True, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    el = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    op = args.out if args.nshards == 1 else args.out.replace(".jsonl", f".{args.shard}.jsonl")
    fails = Counter(); ntok = 0; unread = 0
    with open(op, "w", encoding="utf-8") as f:
        for r, o in zip(rats, outs):
            g = o.outputs[0]; ntok += len(g.token_ids)
            ans, just = parse_answers(g.text)
            bad = [k for k in "abc" if ans[k] is False]
            if any(ans[k] is None for k in "abc"):
                unread += 1
            passed = (not bad) and all(ans[k] is not None for k in "abc")
            for k in bad:
                fails[k] += 1
            f.write(json.dumps({**{k: r[k] for k in
                                   ("doc_id", "hypothesis_id", "hypothesis",
                                    "gold_verdict", "reasoning")},
                                "validator": args.validator,
                                "a": ans["a"], "b": ans["b"], "c": ans["c"],
                                "just_a": just["a"], "just_b": just["b"],
                                "just_c": just["c"],
                                "passed": passed, "fail_reasons": bad,
                                "raw_output": g.text}, ensure_ascii=False) + "\n")
    n = len(rats)
    print(f"\nwrote {op}  ({el/60:.1f} min, {ntok/el:.0f} tok/s, {el/n:.2f}s/row)")
    print(f"  unreadable answers: {unread} ({unread/n:.1%})")
    for k in "abc":
        print(f"  failed ({k}): {fails[k]} ({fails[k]/n:.1%})")


if __name__ == "__main__":
    main()
