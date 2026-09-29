#!/usr/bin/env python3
"""Run BOTH task heads over ONE document so a single review has real
compliance findings AND a real clause inventory.

The evaluation sets keep each contract in one task only, so no archived
prediction covers both. This is 17 positions + 18 categories = 35 prompts on
one contract: a couple of minutes of GPU.

Clause categories returning "not detected" on an NDA is a CORRECT result, not a
gap. The inventory is honest about what the taxonomy finds.
"""
import sys, json, re, time, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CUAD_SYSTEM = (
    "You are a contract review assistant. Given the CONTRACT and a CLAUSE "
    "CATEGORY, determine whether the contract contains a clause of that "
    "category. If it does, return the exact sentence(s) from the contract that "
    "constitute it; if it does not, return an empty evidence list. Respond with "
    'JSON only, in the form {"present": true|false, "evidence": [...]}.'
)
CUAD_USER = "CONTRACT: {text}\n\nCLAUSE CATEGORY: {category}"
RN_SYSTEM = (
    "You are a contract review assistant. Given a clause and its category, "
    "write EXACTLY ONE sentence for a business reader explaining what the "
    "clause does in practice and what brings it into effect."
)
RN_USER = "CLAUSE CATEGORY: {cat}\n\nCLAUSE: {clause}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter",
                    default="/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final")
    ap.add_argument("--doc-id", default="")
    ap.add_argument("--out", default="webapp/data/one_document.json")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    dev = [json.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]
    nli = defaultdict(list)
    for r in dev:
        if r["task"] == "contractnli":
            nli[r["doc_id"]].append(r)

    def q_of(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        m = re.search(r"\n\nQUESTION:\s*(.*)$", u, re.S)
        return m.group(1).strip() if m else ""

    def doc_of(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        m = re.search(r"^CONTRACT:\s*(.*?)\s*\n\nQUESTION:", u, re.S)
        return m.group(1) if m else u

    def pick(k):
        items = nli[k]
        if len({r["verdict"] for r in items}) < 3:
            return -1
        n = len(doc_of(items[0]))
        return (1 if 8000 < n < 40000 else 0) * 100 + len(items)

    cuad = defaultdict(list)
    for r in dev:
        if r["task"] == "cuad":
            cuad[r["contract"]].append(r)
    cats = sorted({r["category"] for r in dev if r["task"] == "cuad"})
    positions = None

    def doc_cuad(r):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        tail = f"\n\nCLAUSE CATEGORY: {r['category']}"
        return u[len("CONTRACT: "):-len(tail)]

    def cpick(c):
        it = cuad[c]
        n = len(doc_cuad(it[0]))
        return (1 if 12000 < n < 45000 else 0) * 1000 + sum(1 for r in it if r["present"])

    nda_id = args.doc_id or max(nli, key=pick)
    nda_items = sorted(nli[nda_id], key=q_of)
    positions = [q_of(r) for r in nda_items]
    com_name = max(cuad, key=cpick)

    DOCS = [
        {"key": "nda", "name": "Mutual Non-Disclosure Agreement",
         "text": doc_of(nda_items[0]), "gold": nda_items},
        {"key": "commercial", "name": "Commercial Agreement",
         "text": doc_cuad(cuad[com_name][0]), "gold": None,
         "cuad_gold": {r["category"]: r for r in cuad[com_name]}},
    ]
    for D in DOCS:
        print(f"[doc] {D['key']}: {len(D['text']):,} chars", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(args.base)

    def enc(msgs):
        e = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    enable_thinking=False)
        if hasattr(e, "input_ids"):
            e = e["input_ids"]
        if e and isinstance(e[0], list):
            e = e[0]
        e = list(e)
        budget = args.max_model_len - args.max_new_tokens
        return e[-budget:] if len(e) > budget else e

    NLI_SYSTEM = [m for m in nda_items[0]["messages"]
                  if m["role"] == "system"][0]["content"]
    prompts, kinds = [], []
    for D in DOCS:
        for qi, q in enumerate(positions):
            prompts.append(enc([
                {"role": "system", "content": NLI_SYSTEM},
                {"role": "user", "content": f"CONTRACT: {D['text']}\n\nQUESTION: {q}"}]))
            kinds.append(("position", D, qi))
        for c in cats:
            prompts.append(enc([{"role": "system", "content": CUAD_SYSTEM},
                                {"role": "user", "content": CUAD_USER.format(
                                    text=D["text"], category=c)}]))
            kinds.append(("clause", D, c))
    print(f"[plan] {len(DOCS)} documents x ({len(positions)} positions + "
          f"{len(cats)} categories) = {len(prompts)} prompts", flush=True)

    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enable_lora=True, max_lora_rank=16, enable_prefix_caching=True,
              enforce_eager=True)
    lr = LoRARequest("adapter", 1, args.adapter)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                        lora_request=lr)
    el = time.time() - t0
    print(f"[gen] {len(prompts)} prompts in {el:.1f}s", flush=True)

    from doc_harness import parse_output, strip_wrappers, _find_json
    per = {D["key"]: [] for D in DOCS}
    notes_needed = []
    for (kind, D, ref), o in zip(kinds, outs):
        raw = o.outputs[0].text or ""
        bag = per[D["key"]]
        if kind == "position":
            d = parse_output(raw)
            gold = (D["gold"][ref]["verdict"] if D.get("gold") else None)
            bag.append({"kind": "position", "position": positions[ref],
                        "predicted": d["verdict"], "gold": gold,
                        "evidence": d["evidence"]})
        else:
            t = strip_wrappers(raw)
            j = _find_json(t) if t else None
            pres = bool(j.get("present")) if isinstance(j, dict) else False
            ev = j.get("evidence") or [] if isinstance(j, dict) else []
            ev = [str(x) for x in ev] if isinstance(ev, list) else []
            g = D.get("cuad_gold", {}).get(ref)
            bag.append({"kind": "clause", "category": ref,
                        "predictedPresent": pres, "evidence": ev,
                        "gold": (bool(g["present"]) if g else None)})
            if pres and ev:
                notes_needed.append((D["key"], len(bag) - 1, ref, " ".join(ev)))

    # risk note for each detected clause, same adapter, second pass
    if notes_needed:
        np_ = [enc([{"role": "system", "content": RN_SYSTEM},
                    {"role": "user", "content": RN_USER.format(cat=c, clause=cl[:4000])}])
               for _, _, c, cl in notes_needed]
        nouts = llm.generate([TokensPrompt(prompt_token_ids=p) for p in np_], sp,
                             lora_request=lr)
        for (k, idx, _, _), o in zip(notes_needed, nouts):
            per[k][idx]["riskNote"] = " ".join((o.outputs[0].text or "").split())
    print(f"[notes] {len(notes_needed)} risk notes generated", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"documents": [{"key": D["key"], "name": D["name"],
                              "documentText": D["text"], "rows": per[D["key"]]}
                             for D in DOCS],
               "base": args.base, "adapter": args.adapter,
               "wall_seconds": round(el, 1)},
              open(args.out, "w"), indent=1)
    for D in DOCS:
        rws = per[D["key"]]
        npos = sum(1 for r in rws if r["kind"] == "position")
        ncl = sum(1 for r in rws if r["kind"] == "clause" and r["predictedPresent"])
        print(f"-> {D['key']}: {npos} positions, {ncl} clauses detected")


if __name__ == "__main__":
    main()
