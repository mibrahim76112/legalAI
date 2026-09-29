#!/usr/bin/env python3
"""B -- sliding-window CUAD inference. Same checkpoint as A, no retraining.

Differences from the earlier v2/window_cuad.py, which this supersedes:

  1. CASE 1 is honoured. A contract is windowed ONLY if the complete prompt
     (system + whole contract + category + generation prompt) exceeds the input
     budget. The earlier script split anything over 14,000 CONTENT tokens, which
     would have windowed a 14.5K contract that actually fits in one call.
  2. Evidence is mapped back to ORIGINAL CONTRACT character offsets. Windows are
     cut on tokenizer offset boundaries and sliced out of the original string,
     so a window is an exact substring of the contract and its char offsets are
     exact rather than reconstructed from decode().

Windowing: ~14,000 content tokens, 2,000 overlap, stride 12,000, final window
extended to the end. Every character of a long contract is covered.

Aggregation: a window's present vote counts only if its emitted span is
grounded in THAT window (reasoning_filters._grounded). Contract-level present
if ANY window casts a grounded vote. No thresholds, no calibration, no voting.
"""
import os, sys, json, time, argparse
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


# ---------------------------------------------------------------- coordinates
def norm_with_map(s):
    """norm(s) plus a per-character map back into s.

    Must agree with text_norm.norm exactly; asserted by the caller. Needed
    because _grounded works on normalized text but coordinates must be
    reported in the ORIGINAL string.
    """
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def locate(span, hay_raw, hay_norm, hay_idx):
    """-> (start, end) char offsets of `span` inside hay_raw, or None.

    Mirrors _grounded's tolerance for the edge punctuation a model adds when
    quoting a heading, so anything _grounded accepts can also be located.
    """
    for cand in (norm(span), norm(span).strip(".,;:!?’'\"-— ")):
        if not cand:
            continue
        p = hay_norm.find(cand)
        if p >= 0:
            return hay_idx[p], hay_idx[p + len(cand) - 1] + 1
    return None


def windows_for(text, offs, n_tok, budget, overhead, content=14000, stride=12000):
    """-> list of dicts with token and CHARACTER ranges into the original text.

    CASE 1: whole contract fits in the budget -> exactly one window covering it.
    CASE 2: overlapping windows on tokenizer offset boundaries.
    """
    if n_tok + overhead <= budget:
        return [{"window": 0, "tok_start": 0, "tok_end": n_tok,
                 "char_start": 0, "char_end": len(text), "windowed": False}]
    out, i, w = [], 0, 0
    while i < n_tok:
        j = min(i + content, n_tok)
        cs = offs[i][0]
        ce = offs[j - 1][1]
        if j >= n_tok:
            ce = len(text)                      # final window runs to the end
        out.append({"window": w, "tok_start": i, "tok_end": j,
                    "char_start": cs, "char_end": ce, "windowed": True})
        if j >= n_tok:
            break
        i += stride
        w += 1
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


def contract_text(r):
    u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
    assert u.startswith("CONTRACT: "), "unexpected CUAD prompt shape"
    tail = f"\n\nCLAUSE CATEGORY: {r['category']}"
    assert u.endswith(tail), "unexpected CUAD prompt tail"
    return u[len("CONTRACT: "):-len(tail)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter",
                    default="/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final")
    ap.add_argument("--data", default="v2/combined/dev.jsonl")
    ap.add_argument("--tag", default="q4_3task_win")
    ap.add_argument("--out-dir", default="/scratch/ibi761/legalai/v2_results")
    ap.add_argument("--content-tokens", type=int, default=14000)
    ap.add_argument("--stride", type=int, default=12000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--cost-only", action="store_true")
    args = ap.parse_args()

    budget = args.max_model_len - args.max_new_tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r.get("task") == "cuad"]
    bycon = defaultdict(list)
    for r in rows:
        bycon[r["contract"]].append(r)
    cats = sorted({r["category"] for r in rows})
    print(f"[data] {len(rows)} CUAD dev rows, {len(bycon)} contracts, "
          f"{len(cats)} categories", flush=True)

    # Scaffolding overhead, measured not guessed. Use the LONGEST category name
    # so one fit decision serves every category of a contract and the windows
    # stay shareable.
    longest = max(cats, key=len)
    probe = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": USER_TMPL.format(text="", category=longest)}]
    enc = tok.apply_chat_template(probe, tokenize=True, add_generation_prompt=True,
                                 enable_thinking=False)
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    overhead = len(list(enc))
    print(f"[fit] budget={budget} scaffold_overhead={overhead} "
          f"(longest category {longest!r}) -> contract fits if <= "
          f"{budget - overhead} tokens", flush=True)

    info = {}
    for cname, rs in bycon.items():
        t = contract_text(rs[0])
        for r in rs[1:]:
            assert contract_text(r) == t, f"contract text differs within {cname}"
        e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        offs = e["offset_mapping"]
        n_tok = len(e["input_ids"])
        nrm, idx = norm_with_map(t)
        assert nrm == norm(t), f"norm_with_map disagrees with norm on {cname}"
        wins = windows_for(t, offs, n_tok, budget, overhead,
                           args.content_tokens, args.stride)
        info[cname] = {"text": t, "n_tok": n_tok, "wins": wins,
                       "norm": nrm, "idx": idx}

    multi = [c for c in info if len(info[c]["wins"]) > 1]
    wc = Counter(len(info[c]["wins"]) for c in info)
    n_calls = sum(len(info[c]["wins"]) * len(bycon[c]) for c in bycon)
    avg_w = sum(len(info[c]["wins"]) for c in info) / len(info)
    print(f"\n[cost] contracts needing MULTIPLE windows : {len(multi)}")
    print(f"[cost] contracts evaluated in ONE window    : {len(info)-len(multi)}")
    print(f"[cost] total model calls for B             : {n_calls:,}")
    print(f"[cost] total model calls for A             : {len(rows):,}")
    print(f"[cost] average windows per contract        : {avg_w:.2f}")
    print(f"[cost] maximum windows for any contract    : {max(wc)}")
    print(f"[cost] windows-per-contract histogram      : {dict(sorted(wc.items()))}")
    cost = {"contracts_multi_window": len(multi),
            "contracts_single_window": len(info) - len(multi),
            "n_calls_B": n_calls, "n_calls_A": len(rows),
            "avg_windows_per_contract": avg_w, "max_windows": max(wc),
            "histogram": {str(k): v for k, v in sorted(wc.items())},
            "budget": budget, "scaffold_overhead": overhead,
            "contract_tokens": {c: info[c]["n_tok"] for c in info}}
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    json.dump(cost, open(outd / f"cost__{args.tag}.json", "w"), indent=2)
    if args.cost_only:
        return

    # -------------------------------------------------------------- generation
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    jobs, prompts, n_over = [], [], 0
    for cname, rs in bycon.items():
        I = info[cname]
        for w in I["wins"]:
            wtext = I["text"][w["char_start"]:w["char_end"]]
            for r in rs:
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER_TMPL.format(
                            text=wtext, category=r["category"])}]
                e = tok.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
                if hasattr(e, "input_ids"):
                    e = e["input_ids"]
                if e and isinstance(e[0], list):
                    e = e[0]
                e = list(e)
                if len(e) > budget:
                    n_over += 1
                    e = e[-budget:]
                prompts.append(e)
                jobs.append({"contract": cname, "category": r["category"],
                             "window": w["window"], "windowed": w["windowed"],
                             "win_char_start": w["char_start"],
                             "win_char_end": w["char_end"],
                             "win_tok_start": w["tok_start"],
                             "win_tok_end": w["tok_end"],
                             "gold_present": bool(r.get("present")),
                             "gold_evidence": r.get("evidence") or []})
    print(f"\n[gen] {len(prompts):,} prompts ({n_over} over budget after "
          f"windowing -- should be 0)", flush=True)

    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enable_lora=True, max_lora_rank=16, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    lr = LoRARequest("adapter", 1, args.adapter)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                        lora_request=lr)
    el = time.time() - t0
    print(f"[gen] {el/60:.1f} min, {el/len(prompts):.3f}s/call", flush=True)

    # ------------------------------------------- parse, ground, map coordinates
    wcache = {}
    wfp = outd / f"windows__{args.tag}.jsonl"
    map_fail = 0
    with open(wfp, "w", encoding="utf-8") as f:
        for j, o in zip(jobs, outs):
            raw = o.outputs[0].text or ""
            p, ev = parse(raw)
            I = info[j["contract"]]
            key = (j["contract"], j["window"])
            if key not in wcache:
                wt = I["text"][j["win_char_start"]:j["win_char_end"]]
                wn, wi = norm_with_map(wt)
                wcache[key] = (wt, wn, wi)
            wt, wn, wi = wcache[key]
            spans = []
            for s in ev:
                if not _grounded(s, wn):
                    continue
                loc = locate(s, wt, wn, wi)
                if loc is None:
                    map_fail += 1
                    continue
                ls, le = loc
                spans.append({"text": s,
                              "window_char_start": ls, "window_char_end": le,
                              "contract_char_start": j["win_char_start"] + ls,
                              "contract_char_end": j["win_char_start"] + le})
            j.update({"raw": raw, "pred_present": p, "pred_evidence": ev,
                      "grounded_spans": spans,
                      "n_ungrounded": len(ev) - len(spans),
                      "vote_present": bool(p and spans)})
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    json.dump({"tag": args.tag, "base": args.base, "adapter": args.adapter,
               "adapter_sha256_note": "see report; verified against A",
               "data": args.data, "content_tokens": args.content_tokens,
               "stride": args.stride, "max_model_len": args.max_model_len,
               "max_new_tokens": args.max_new_tokens, "input_budget": budget,
               "n_calls": len(prompts), "wall_seconds": round(el, 1),
               "sec_per_call": round(el / len(prompts), 3),
               "grounded_but_unmappable_spans": map_fail, **cost},
              open(outd / f"windowmeta__{args.tag}.json", "w"), indent=2)
    print(f"[map] grounded but unmappable spans: {map_fail}")
    print(f"-> {wfp}")


if __name__ == "__main__":
    main()
