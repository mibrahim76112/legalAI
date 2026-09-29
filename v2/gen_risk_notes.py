#!/usr/bin/env python3
"""
Task 3 — synthetic risk notes.

Source clause: expert-annotated CUAD evidence spans from the 18 selected
categories. The CLAUSE is human-annotated; only the NOTE is synthetic.

Split inheritance: a note inherits the split of the CUAD contract its clause
came from, so a contract in CUAD train cannot contribute a note to risk-note
test. Enforced by construction -- the split is copied from the source row.

Filters (all reused, none invented):
  1. single sentence
  2. length bound
  3. groundedness -- reasoning_filters._grounded, which wraps text_norm.norm
"""
import os, sys, json, time, argparse, re
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                      # noqa: E402

SEED = 20260919
# The first version of this prompt said "explains why this clause matters to
# the business". Combined with temperature=0.0, the teacher echoed that phrase
# verbatim: 85% of a 40-row smoke opened with the identical six words and there
# were 5 distinct four-word openings across 40 notes. The three filters
# (sentence count, length, groundedness) are all blind to templating, so the
# batch passed at 100%. The prompt no longer supplies a stem to copy, and the
# banned opening is stated explicitly.
SYSTEM = (
    "You are a contract review assistant. A business reader with no legal "
    "training is about to sign a contract and has been shown one clause from "
    "it.\n\n"
    "In EXACTLY ONE sentence, tell that reader two things: what the clause "
    "does to them in practice, and what event or condition brings it into "
    "effect.\n\n"
    "Vary your phrasing between notes; do not fall into a fixed formula. Do "
    "NOT begin with the words 'This clause'. Do not recommend legal action. "
    "Do not suggest negotiating, amending, or consulting counsel. Do not state "
    "any fact that is not supported by the clause text itself. Write one "
    "sentence and nothing else -- no preamble, no bullet points, no quotation "
    "marks around the whole sentence."
)
USER = "CLAUSE CATEGORY: {cat}\n\nCLAUSE: {clause}"


def ids(enc):
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


_SENT = re.compile(r"[.!?](?:\s|$)")


def sentence_count(t):
    """Count terminal punctuation, ignoring common abbreviations and decimals."""
    s = re.sub(r"\b(?:e\.g|i\.e|etc|vs|Inc|Ltd|Corp|Co|No|Art|Sec)\.", "X", t or "")
    s = re.sub(r"\d\.\d", "XX", s)
    return len(_SENT.findall(s.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3-32B")
    ap.add_argument("--cuad-dir", default="v2/cuad")
    ap.add_argument("--out", default="v2/risk_notes")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-same-opening-frac", type=float, default=0.05,
                    help="max share of kept notes allowed to have one 5-word "
                         "opening (f4); a fixed count would be far too "
                         "permissive on a small smoke and too strict at scale")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # one note per PRESENT row; the row's spans are the source clause
    rows = []
    for sp in ("train", "dev", "test"):
        for l in open(Path(args.cuad_dir) / f"{sp}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["present"] and r["evidence"]:
                rows.append(r)
    rows.sort(key=lambda r: (r["contract"], r["category"]))
    if args.limit:
        rows = rows[:args.limit]
    print(f"source rows (present=true): {len(rows)}")
    print(f"  split inheritance: {dict(Counter(r['split'] for r in rows))}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(args.teacher)

    def think_kw():
        m = [{"role": "user", "content": "x"}]
        for n in ("enable_thinking", "thinking"):
            try:
                if tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                           **{n: True}) != \
                   tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                           **{n: False}):
                    return n
            except Exception:
                pass
        return None
    kw = think_kw()
    extra = {kw: False} if kw else {}

    prompts = []
    for r in rows:
        clause = " ".join(" ".join(r["evidence"]).split())[:4000]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(cat=r["category"], clause=clause)}]
        prompts.append(ids(tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, **extra)))
    plen = sorted(len(p) for p in prompts)
    print(f"  prompt tokens p50={plen[len(plen)//2]} max={plen[-1]}", flush=True)

    llm = LLM(model=args.teacher, dtype="bfloat16", tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, enable_prefix_caching=True)
    # Greedy decoding was the second half of the templating problem: one system
    # prompt plus argmax gives the same opening every time. Sampling restores
    # variety, and a per-request seed derived from SEED keeps the run exactly
    # reproducible -- the seed is recorded in the manifest.
    sps = [SamplingParams(temperature=args.temperature, top_p=args.top_p,
                          max_tokens=args.max_new_tokens, seed=SEED + i)
           for i in range(len(prompts))]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps)
    el = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)

    # ---- length bound derived from the observed distribution, not hardcoded ----
    words = sorted(len((o.outputs[0].text or "").split()) for o in outs)
    lo = max(8, words[int(0.01 * len(words))])
    hi = words[min(len(words) - 1, int(0.99 * len(words)))]
    print(f"\nlength bounds from observed p1/p99 (floor 8): {lo}..{hi} words", flush=True)

    # f4 cap scales with the run: 5% of rows, floor 2.
    open_cap = max(2, int(args.max_same_opening_frac * len(rows)))
    print(f"f4 templating cap: {open_cap} notes may share a 5-word opening "
          f"({args.max_same_opening_frac:.0%} of {len(rows)})", flush=True)
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    kept, rej = [], Counter()
    opening_ct = Counter()
    with open(outd / "all_raw.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            note = " ".join((o.outputs[0].text or "").strip().split())
            clause = " ".join(r["evidence"])
            hay = norm(clause)
            why = []
            ns = sentence_count(note)
            if ns != 1:
                why.append(f"f1_sentences({ns})")
            w = len(note.split())
            if w < lo:
                why.append(f"f2_too_short({w})")
            elif w > hi:
                why.append(f"f2_too_long({w})")
            # groundedness: any quoted span in the note must occur in the clause
            q = re.findall(r'"([^"]{8,})"', note)
            ung = [x for x in q if norm(x) not in hay]
            if ung:
                why.append(f"f3_ungrounded_quote({len(ung)})")
            # f4 -- templating. The first three filters are blind to it: a
            # note can be one sentence, the right length, and fully grounded
            # while being the 200th copy of the same sentence stem. Training on
            # that teaches a template rather than reasoning, so repeats of any
            # 5-word opening are capped.
            open5 = " ".join(note.lower().split()[:5])
            if not why and opening_ct[open5] >= open_cap:
                why.append(f"f4_templated_opening")
            if not why:
                opening_ct[open5] += 1
            rec = {"contract": r["contract"], "category": r["category"],
                   "split": r["split"], "clause": clause, "note": note,
                   "n_words": w, "n_sentences": ns,
                   "teacher": args.teacher, "reject_reasons": why,
                   "kept": not why,
                   "n_output_tokens": len(o.outputs[0].token_ids)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (kept.append(rec) if not why else rej.update([why[0]]))

    for sp in ("train", "dev", "test"):
        sub = [k for k in kept if k["split"] == sp]
        with open(outd / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for k in sub:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"\ngenerated {n} in {el/60:.1f} min ({ntok/el:.0f} tok/s, {el/n:.2f}s/row)")
    print(f"kept {len(kept)}/{n} = {len(kept)/n:.1%}")
    if kept:
        oc = Counter(" ".join(k["note"].lower().split()[:5]) for k in kept)
        top, topn = oc.most_common(1)[0]
        print(f"diversity: {len(oc)} distinct 5-word openings over {len(kept)} "
              f"kept; most common {topn} ({topn/len(kept):.1%}) = {top!r}")
    for k, v in rej.most_common():
        print(f"  rejected {k:28s} {v:5d} ({v/n:.1%})")
    json.dump({"teacher": args.teacher, "teacher_license": "apache-2.0",
               "seed": SEED, "source_rows": n, "kept": len(kept),
               "length_bounds_words": [lo, hi],
               "reject_reasons": dict(rej),
               "diversity": {
                   "distinct_5word_openings": len({" ".join(k["note"].lower().split()[:5])
                                                   for k in kept}),
                   "most_common_opening": (Counter(
                       " ".join(k["note"].lower().split()[:5]) for k in kept
                   ).most_common(1)[0] if kept else None),
                   "max_same_opening_allowed": open_cap,
                   "max_same_opening_frac": args.max_same_opening_frac},
               "kept_by_split": {sp: sum(1 for k in kept if k["split"] == sp)
                                 for sp in ("train", "dev", "test")},
               "wall_seconds": round(el, 1), "tokens_per_sec": round(ntok / el, 1)},
              open(outd / "_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
