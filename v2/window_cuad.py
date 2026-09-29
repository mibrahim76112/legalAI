#!/usr/bin/env python3
"""Windowed CUAD inference. Eval-time only -- no retraining, same checkpoints.

Truncated inference keeps the LAST 15,360 tokens of a contract. Any gold span
before that cut is invisible, which caps evidence recall at ~89.9% on dev
through no fault of the model. Windowing covers the whole contract instead.

Windows are 14,000 CONTENT tokens with 2,000 overlap (stride 12,000). The
overlap exists so a clause straddling a boundary is intact in at least one
window. Budget is 15,360, leaving ~1,300 for the system prompt, the
"CONTRACT:"/"CLAUSE CATEGORY:" scaffolding and the category name.

AGGREGATION (fixed in advance, no calibration, no tuned threshold):
  A window's `present: true` counts ONLY if the span it emitted is grounded in
  THAT window's text. A window claiming presence with a span that is not in the
  text it was shown is hallucinating, not detecting -- and that is exactly the
  false-positive class that max-pooling over N windows amplifies. Requiring
  groundedness is what makes OR-over-windows safe.
  Contract-level: present if ANY window votes present with a grounded span.
"""
import os, sys, json, re, time, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402
from doc_harness import strip_wrappers, _find_json             # noqa: E402

SYSTEM = (
    "You are a contract review assistant. Given the CONTRACT and a CLAUSE "
    "CATEGORY, determine whether the contract contains a clause of that "
    "category. If it does, return the exact sentence(s) from the contract that "
    "constitute it; if it does not, return an empty evidence list. Respond with "
    'JSON only, in the form {"present": true|false, "evidence": [...]}.'
)
USER_TMPL = "CONTRACT: {text}\n\nCLAUSE CATEGORY: {category}"


def contract_text(r):
    """Recover the contract text from the built prompt. Exact, not heuristic:
    build_cuad.py formats it as CONTRACT: {text}\\n\\nCLAUSE CATEGORY: {cat}."""
    u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
    assert u.startswith("CONTRACT: "), "unexpected CUAD prompt shape"
    tail = f"\n\nCLAUSE CATEGORY: {r['category']}"
    assert u.endswith(tail), "unexpected CUAD prompt tail"
    return u[len("CONTRACT: "):-len(tail)]


def windows_of(tok, text, content=14000, stride=12000):
    """-> list of (start_tok, text). Whole contract covered; last window is
    shortened rather than padded."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= content:
        return [(0, text)]
    out, i = [], 0
    while i < len(ids):
        chunk = ids[i:i + content]
        out.append((i, tok.decode(chunk)))
        if i + content >= len(ids):
            break
        i += stride
    return out


def parse(raw):
    t = strip_wrappers(raw)
    if not t:
        return None, []
    d = _find_json(t)
    if not isinstance(d, dict) or "present" not in d:
        return None, []
    ev = d.get("evidence") or []
    if not isinstance(ev, list):
        ev = []
    return bool(d["present"]), [str(x) for x in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="v2/combined/dev.jsonl")
    ap.add_argument("--tag", default="windowed")
    ap.add_argument("--out-dir", default="/scratch/ibi761/legalai/v2_results")
    ap.add_argument("--content-tokens", type=int, default=14000)
    ap.add_argument("--stride", type=int, default=12000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--contracts", type=int, default=0,
                    help="limit to N contracts (stratified by length)")
    ap.add_argument("--cost-only", action="store_true",
                    help="report window/call counts and exit, no model load")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")
            if '"task": "cuad"' in l or json.loads(l).get("task") == "cuad"]
    rows = [r for r in rows if r.get("task") == "cuad"]
    bycon = defaultdict(list)
    for r in rows:
        bycon[r["contract"]].append(r)
    print(f"[data] {len(rows)} CUAD dev rows over {len(bycon)} contracts", flush=True)

    # window each contract once; the windows are shared by all its categories
    texts, wins, clen = {}, {}, {}
    for cname, rs in bycon.items():
        t = contract_text(rs[0])
        for r in rs[1:]:
            assert contract_text(r) == t, f"contract text differs within {cname}"
        texts[cname] = t
        clen[cname] = len(tok(t, add_special_tokens=False)["input_ids"])
        wins[cname] = windows_of(tok, t, args.content_tokens, args.stride)

    wc = Counter(len(v) for v in wins.values())
    n_calls = sum(len(wins[c]) * len(bycon[c]) for c in bycon)
    print(f"\n[cost] windows per contract: {dict(sorted(wc.items()))}")
    print(f"[cost] contract tokens: min={min(clen.values()):,} "
          f"p50={sorted(clen.values())[len(clen)//2]:,} max={max(clen.values()):,}")
    print(f"[cost] generation calls: windowed={n_calls:,} vs truncated={len(rows):,} "
          f"({n_calls/len(rows):.2f}x)")
    print(f"[cost] NOTE: the system prompt + CONTRACT window is a shared PREFIX "
          f"across all {len(bycon[list(bycon)[0]])} categories of a contract, so "
          f"prefix caching should make the real cost far below {n_calls/len(rows):.2f}x.")

    buckets = Counter()
    for c, n in clen.items():
        buckets["<16K" if n < 16000 else "16-32K" if n < 32000 else ">32K"] += 1
    print(f"[cost] contracts by length: {dict(buckets)}")
    if args.cost_only:
        json.dump({"windows_per_contract": {str(k): v for k, v in wc.items()},
                   "n_calls_windowed": n_calls, "n_calls_truncated": len(rows),
                   "contract_tokens": clen,
                   "length_buckets": dict(buckets)},
                  open(Path(args.out_dir) / f"cost__{args.tag}.json", "w"), indent=2)
        return

    if args.contracts:
        order = sorted(bycon, key=lambda c: clen[c])
        step = max(1, len(order) // args.contracts)
        keep = set(order[::step][:args.contracts])
        bycon = {c: v for c, v in bycon.items() if c in keep}
        n_calls = sum(len(wins[c]) * len(bycon[c]) for c in bycon)
        print(f"[subset] {len(bycon)} contracts, {n_calls:,} calls", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    kw = {}
    probe = [{"role": "user", "content": "x"}]
    for name in ("enable_thinking", "thinking"):
        try:
            a = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **{name: True})
            b = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **{name: False})
        except Exception:
            continue
        if a != b:
            kw = {name: False}
            break

    # Order matters: group by (contract, window) so the shared prefix is hot.
    jobs, prompts = [], []
    budget = args.max_model_len - args.max_new_tokens
    n_over = 0
    for cname, rs in bycon.items():
        for wi, (wstart, wtext) in enumerate(wins[cname]):
            for r in rs:
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER_TMPL.format(
                            text=wtext, category=r["category"])}]
                enc = tok.apply_chat_template(msgs, tokenize=True,
                                              add_generation_prompt=True, **kw)
                if hasattr(enc, "input_ids"):
                    enc = enc["input_ids"]
                if enc and isinstance(enc[0], list):
                    enc = enc[0]
                enc = list(enc)
                if len(enc) > budget:
                    n_over += 1
                    enc = enc[-budget:]
                prompts.append(enc)
                jobs.append({"contract": cname, "category": r["category"],
                             "window": wi, "window_start_tok": wstart,
                             "present_gold": bool(r.get("present")),
                             "gold_evidence": r.get("evidence") or []})
    print(f"[gen] {len(prompts):,} prompts ({n_over} still over budget)", flush=True)

    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enable_lora=bool(args.adapter), max_lora_rank=16,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    lr = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, lora_request=lr)
    el = time.time() - t0
    print(f"[gen] {el/60:.1f} min for {len(prompts):,} calls "
          f"({el/len(prompts):.3f}s/call)", flush=True)

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    wfp = outd / f"windows__{args.tag}.jsonl"
    norm_win = {}
    with open(wfp, "w", encoding="utf-8") as f:
        for j, o in zip(jobs, outs):
            raw = o.outputs[0].text or ""
            p, ev = parse(raw)
            key = (j["contract"], j["window"])
            if key not in norm_win:
                norm_win[key] = norm(wins[j["contract"]][j["window"]][1])
            hay = norm_win[key]
            grounded = [e for e in ev if _grounded(e, hay)]
            j.update({"raw": raw, "pred_present": p,
                      "pred_evidence": ev, "grounded_evidence": grounded,
                      "vote_present": bool(p and grounded)})
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    json.dump({"tag": args.tag, "base": args.base, "adapter": args.adapter or None,
               "content_tokens": args.content_tokens, "stride": args.stride,
               "n_contracts": len(bycon), "n_calls": len(prompts),
               "n_calls_truncated_equivalent": sum(len(v) for v in bycon.values()),
               "wall_seconds": round(el, 1),
               "sec_per_call": round(el / len(prompts), 3),
               "windows_per_contract": {str(k): v for k, v in wc.items()},
               "contract_tokens": {c: clen[c] for c in bycon}},
              open(outd / f"windowmeta__{args.tag}.json", "w"), indent=2)
    print(f"-> {wfp}")


if __name__ == "__main__":
    main()
