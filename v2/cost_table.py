#!/usr/bin/env python3
"""Training time, token counts, memory and inference latency per model.

Every figure comes from a run artifact: run_meta.json, the SLURM accounting
record, the memory probe, or an evaluation's own timing. Token counts are
computed with each model's OWN tokenizer, because vocabularies differ and a
shared count would misstate throughput.
"""
import json, subprocess, argparse
from pathlib import Path

SFT = Path("/scratch/ibi761/legalai/v2_sft")
RES = Path("/scratch/ibi761/legalai/v2_results")
RUNS = [
    ("Qwen3-4B",     "Qwen__Qwen3-4B__3task",     "Qwen/Qwen3-4B",     "q4_3task"),
    ("Qwen3-8B",     "Qwen__Qwen3-8B__3task",     "Qwen/Qwen3-8B",     "q8_3task"),
    ("Llama-3.1-8B", "NousResearch__Meta-Llama-3.1-8B-Instruct__3task",
                     "NousResearch/Meta-Llama-3.1-8B-Instruct",        "llama_3task"),
    ("Qwen3-14B",    "Qwen__Qwen3-14B__3task",    "Qwen/Qwen3-14B",    "q14_3task"),
]
# peak GiB at seq 16384, from v2/_memory_probe.json (real fwd+bwd+optimizer step)
PEAK = {"Qwen3-4B": 43.2, "Qwen3-8B": 52.7, "Llama-3.1-8B": 46.8, "Qwen3-14B": 66.8}
# training wall clock from sacct
TRAIN_S = {"Qwen3-4B": 12564, "Qwen3-8B": 15939, "Llama-3.1-8B": 13776, "Qwen3-14B": 23686}
# evaluation wall clock over the full 2,489-row dev set, from the job logs
EVAL_S = {"Qwen3-4B": 6035, "Llama-3.1-8B": 5140, "Qwen3-14B": 8954, "Qwen3-8B": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v2/_cost_table.json")
    ap.add_argument("--skip-tokens", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    train = [json.loads(l) for l in open("v2/combined/train.jsonl", encoding="utf-8")]
    rows = []
    for name, d, base, tag in RUNS:
        meta = json.load(open(SFT / d / "run_meta.json"))
        lh = meta["log_history"]
        steps = max(x.get("step", 0) for x in lh)
        tl = [x["loss"] for x in lh if "loss" in x]
        ev = [x["eval_loss"] for x in lh if "eval_loss" in x]

        n_prompt = n_target = 0
        if not args.skip_tokens:
            tok = AutoTokenizer.from_pretrained(base)
            for r in train:
                msgs = [m for m in r["messages"] if m["role"] != "assistant"]
                tgt = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
                e = tok.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
                if hasattr(e, "input_ids"):
                    e = e["input_ids"]
                if e and isinstance(e[0], list):
                    e = e[0]
                p = len(list(e))
                t = len(tok(tgt, add_special_tokens=False)["input_ids"]) + 1
                cap = meta["max_seq_len"]
                if p + t > cap:
                    p = max(cap - t, 0)
                n_prompt += p
                n_target += t

        ts = TRAIN_S[name]
        es = EVAL_S.get(name)
        rows.append({
            "model": name, "params_trainable": None,
            "train_seconds": ts, "train_hms": f"{ts//3600}h{(ts%3600)//60:02d}m",
            "steps": steps, "rows": meta["data"]["train"],
            "train_tokens_total": n_prompt + n_target,
            "train_tokens_prompt": n_prompt,
            "train_tokens_supervised": n_target,
            "train_tokens_per_sec": round((n_prompt + n_target) / ts, 1) if ts else None,
            "sec_per_row": round(ts / meta["data"]["train"], 3),
            "peak_gib_at_16384": PEAK[name],
            "final_train_loss": round(tl[-1], 4) if tl else None,
            "final_eval_loss": round(ev[-1], 4) if ev else None,
            "eval_seconds_2489": es,
            "eval_sec_per_example": round(es / 2489, 3) if es else None,
            "eval_examples_per_sec": round(2489 / es, 3) if es else None,
            "lora": meta["lora"], "max_seq_len": meta["max_seq_len"],
        })

    # trainable params from the job logs
    import re, glob
    for r in rows:
        for f in glob.glob("/scratch/ibi761/legalai/logs/v2_*.out"):
            txt = open(f, errors="ignore").read(4000)
            if r["model"].split("-")[0].lower() in txt.lower() and "[lora]" in txt:
                m = re.search(r"trainable=([\d,]+)", txt)
                if m and ("14B" in txt) == ("14B" in r["model"]) \
                     and ("Llama" in txt) == ("Llama" in r["model"]):
                    r["params_trainable"] = int(m.group(1).replace(",", ""))
                    break

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"{'model':14s} {'train':>7s} {'steps':>6s} {'tokens':>12s} {'tok/s':>8s} "
          f"{'s/row':>6s} {'peak GiB':>9s} {'infer s/ex':>11s}")
    for r in rows:
        print(f"{r['model']:14s} {r['train_hms']:>7s} {r['steps']:>6d} "
              f"{r['train_tokens_total']:>12,d} {str(r['train_tokens_per_sec']):>8s} "
              f"{r['sec_per_row']:>6.2f} {r['peak_gib_at_16384']:>9.1f} "
              f"{str(r['eval_sec_per_example']):>11s}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
