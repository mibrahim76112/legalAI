#!/usr/bin/env python3
"""
Held-out generalisation benchmark: LegalBench CUAD tasks.

WHY CUAD AND NOT contract_nli_*: LegalBench contains 14 `contract_nli_*` tasks
built from the SAME source as our training data. Those are NOT held out and
benchmarking on them would not support a generalisation claim. The 38 `cuad_*`
tasks come from the Contract Understanding Atticus Dataset -- different
contracts, different annotators, a different clause taxonomy -- and nothing from
CUAD was used in training.

Uses LegalBench's OWN instruction per task, verbatim from task_metadata.json,
with its own scoring rule (`contained_in_output`). That keeps this the
benchmark's protocol rather than a reformatting of ours, which would otherwise
measure my prompt engineering instead of the model.

The comparison that matters is base vs fine-tuned under an identical prompt: if
the fine-tune is worse, narrow SFT on ContractNLI damaged general legal ability
(catastrophic forgetting). If equal or better, it generalised.
"""
import os, json, csv, time, argparse, random, re
from pathlib import Path
from collections import Counter, defaultdict


def load_env(p=".env"):
    f = Path(p)
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch_tasks(limit_per_task, seed=0):
    from huggingface_hub import hf_hub_download, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files("nguha/legalbench", repo_type="dataset")
    tasks = sorted({f.split("/")[1] for f in files if f.startswith("data/cuad_")})
    meta = json.load(open(hf_hub_download("nguha/legalbench", "task_metadata.json",
                                          repo_type="dataset"), encoding="utf-8"))
    rng = random.Random(seed)
    out = []
    for t in tasks:
        try:
            p = hf_hub_download("nguha/legalbench", f"data/{t}/test.tsv",
                                repo_type="dataset")
        except Exception:
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))
        if limit_per_task and len(rows) > limit_per_task:
            rows = rng.sample(rows, limit_per_task)
        instr = meta[t]["instruction"]
        space = meta[t]["answer_space"]
        for r in rows:
            out.append({"task": t, "text": r["text"], "gold": r["answer"],
                        "instruction": instr, "answer_space": space,
                        "document_name": r.get("document_name", "")})
    return out, tasks


def render(r):
    """LegalBench instruction verbatim, {{text}} substituted."""
    return r["instruction"].replace("{{text}}", r["text"])


def score(out, space):
    """LegalBench `contained_in_output`: first answer-space label appearing in
    the output, matched case-insensitively on a word boundary."""
    t = (out or "")
    hits = []
    for a in space:
        m = re.search(rf"\b{re.escape(a)}\b", t, re.I)
        if m:
            hits.append((m.start(), a))
    return min(hits)[1] if hits else None


def ids(enc):
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--limit-per-task", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out-dir", default=os.environ.get("LB_DIR", "legalbench_results"))
    args = ap.parse_args()
    load_env()

    rows, tasks = fetch_tasks(args.limit_per_task)
    print(f"CUAD tasks: {len(tasks)}   rows: {len(rows)}   "
          f"(<= {args.limit_per_task}/task)")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(args.model)

    def think_kw():
        m = [{"role": "user", "content": "x"}]
        base = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
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

    prompts = [ids(tok.apply_chat_template(
        [{"role": "user", "content": render(r)}], tokenize=True,
        add_generation_prompt=True, **extra)) for r in rows]
    plen = sorted(len(p) for p in prompts)
    keep = [i for i, p in enumerate(prompts)
            if len(p) <= args.max_model_len - args.max_new_tokens]
    if len(keep) < len(prompts):
        print(f"  dropping {len(prompts)-len(keep)} rows over context budget")
    rows = [rows[i] for i in keep]; prompts = [prompts[i] for i in keep]
    print(f"  prompt tokens p50={plen[len(plen)//2]} max={plen[-1]}")

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, trust_remote_code=True,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    el = time.time() - t0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    lab = args.label or args.model.replace("/", "__")
    raw = Path(args.out_dir) / f"lb__{lab}.jsonl"
    per = defaultdict(lambda: [0, 0]); nofmt = 0
    with open(raw, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            pred = score(g.text, r["answer_space"])
            ok = (pred is not None and pred.lower() == r["gold"].lower())
            per[r["task"]][0] += ok; per[r["task"]][1] += 1
            nofmt += (pred is None)
            f.write(json.dumps({"model": args.model, "label": lab, "task": r["task"],
                                "gold": r["gold"], "pred": pred,
                                "raw_output": g.text,
                                "n_output_tokens": len(g.token_ids),
                                "document_name": r["document_name"]},
                               ensure_ascii=False) + "\n")
    n = sum(v[1] for v in per.values()); c = sum(v[0] for v in per.values())
    macro = sum(v[0]/v[1] for v in per.values())/len(per)
    print(f"\nwrote {raw}  ({el/60:.1f} min)")
    print(f"  micro accuracy {c}/{n} = {c/n:.1%}")
    print(f"  macro over {len(per)} tasks = {macro:.1%}")
    print(f"  unparseable (no Yes/No in output) = {nofmt/n:.1%}")
    json.dump({"model": args.model, "label": lab, "micro": c/n, "macro": macro,
               "n": n, "tasks": len(per), "unparseable": nofmt/n,
               "per_task": {k: v[0]/v[1] for k, v in per.items()},
               "wall_seconds": round(el, 1)},
              open(Path(args.out_dir)/f"lbmeta__{lab}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
