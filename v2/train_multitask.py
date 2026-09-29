#!/usr/bin/env python3
"""LoRA SFT on the combined multi-task dataset.

Why this exists rather than reusing v1_contractnli/train_sft.py: that script
does `ex["verdict"] = r["verdict"]` unconditionally, and CUAD rows have no
verdict field -- they carry `present: bool`. v1 is frozen, so it is read, not
edited. Everything that CAN be reused is imported from it: build_example, the
collator contract, and the same LoRA/optimizer settings.

Deliberately simpler than v1's trainer: no Contradiction oversampling, no class
weighting, no reasoning-token weighting. Those were levers for a single-task
precision/recall trade and reintroducing them here would confound the
multi-task question with a rebalancing question.
"""
import os, sys, json, argparse, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from doc_sft_seq import (build_example, detect_think_kwarg,   # noqa: E402
                         IGNORE_INDEX)
from model_loading import load_model                          # noqa: E402


def label_of(r):
    """One stratification label per row, whatever the task."""
    t = r.get("task")
    if t == "contractnli":
        return f"nli:{r['verdict']}"
    if t == "risknote":
        return f"risknote:{r.get('category', '?')}"
    return f"cuad:{'present' if r.get('present') else 'absent'}"


class MultiTaskSFT(Dataset):
    def __init__(self, path, tok, think_kwarg, max_seq_len, limit=0):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if limit:
            rows = rows[:limit]
        self.n_trunc = self.n_dropped = 0
        self.items = []
        for r in rows:
            ex = build_example(tok, r["messages"], think_kwarg, max_seq_len)
            if ex is None:
                self.n_dropped += 1
                continue
            self.n_trunc += ex["truncated"]
            ex["task"] = r.get("task", "unknown")
            ex["label"] = label_of(r)
            self.items.append(ex)
        self.task_counts = dict(Counter(e["task"] for e in self.items))
        self.label_counts = dict(Counter(e["label"] for e in self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e = self.items[i]
        return {"input_ids": e["input_ids"], "labels": e["labels"]}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        lab[i, :L] = torch.tensor(b["labels"], dtype=torch.long)
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="v2/combined/train.jsonl")
    ap.add_argument("--valid", default="v2/combined/dev.jsonl")
    ap.add_argument("--out-dir", default=os.environ.get("SFT_DIR", "/scratch/ibi761/legalai/v2_sft"))
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--variant", default="")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA: freeze the base in NF4 4-bit and train LoRA "
                         "on top. Compare against the bf16 LoRA run to "
                         "separate the cost of quantizing DURING TRAINING from "
                         "the cost of quantizing at inference.")
    ap.add_argument("--bnb-compute-dtype", default="bfloat16")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # build_example expects the kwarg NAME (a string), not a dict.
    probe = [{"role": "user", "content": "x"}]
    kw, default_on = detect_think_kwarg(tok, probe)
    print(f"[tmpl] think_kwarg={kw!r} template_default_is_thinking={default_on}")

    tr = MultiTaskSFT(args.train, tok, kw, args.max_seq_len, args.limit)
    va = MultiTaskSFT(args.valid, tok, kw, args.max_seq_len, args.eval_n)
    print(f"[data] train={len(tr)} (trunc {tr.n_trunc}, dropped {tr.n_dropped}) "
          f"valid={len(va)}")
    print(f"[data] train tasks={tr.task_counts}")
    print(f"[data] train labels={tr.label_counts}")

    # Prove the loss mask before spending GPU hours on it.
    e = tr.items[0]
    n_prompt = e["n_prompt"]
    assert all(x == IGNORE_INDEX for x in e["labels"][:n_prompt]), \
        "loss mask broken: prompt tokens are not -100"
    assert any(x != IGNORE_INDEX for x in e["labels"][n_prompt:]), \
        "loss mask broken: target is entirely -100"
    print(f"[mask] ok -- first {n_prompt} labels are -100, "
          f"{sum(1 for x in e['labels'] if x != IGNORE_INDEX)} supervised tokens")

    if args.load_in_4bit:
        # QLoRA. nf4 + double quantization is the Dettmers et al. recipe; the
        # compute dtype stays bf16 so only STORAGE is 4-bit, which is the
        # variable under test. prepare_model_for_kbit_training casts norms and
        # the head back to fp32 and re-enables input grads under checkpointing.
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, args.bnb_compute_dtype))
        model = load_model(args.model, torch.bfloat16, quantization_config=bnb)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
        print("[quant] base frozen in NF4 4-bit, compute dtype "
              f"{args.bnb_compute_dtype}")
    else:
        # gemma-4-12B-it is Gemma4UnifiedForConditionalGeneration, not a plain
        # CausalLM; v1's loader tries the conditional-generation classes too.
        model = load_model(args.model, torch.bfloat16)
    # NemotronH: transformers 5.14 builds a DynamicCache from config.layer_types,
    # which for nemotron_h contains 27 'linear_attention', 25 'mlp' and 4
    # 'full_attention' entries. DYNAMIC_LAYER_TYPE_MAPPING has no 'mlp' key, so
    # the forward dies with KeyError: 'mlp'. The construction is guarded by
    # `if use_cache and past_key_values is None`, so disabling the cache avoids
    # it entirely -- and the cache is useless during checkpointed training
    # anyway. Set explicitly rather than relying on
    # gradient_checkpointing_enable() to do it.
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            try:
                model.config.text_config.use_cache = False
            except Exception:
                pass
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # Candidate projections. in_proj/out_proj are the Mamba mixer projections in
    # hybrid models such as NemotronH: without them a LoRA would adapt only the
    # 4 attention and 25 MLP layers and leave all 27 mixers untouched, which is
    # not the same intervention the dense models get and would make a weak
    # result an artifact of adapter placement rather than of the model.
    # Names absent from a given architecture are filtered out, so dense models
    # are unaffected. The resolved list is printed and recorded in run_meta.
    tgts = ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj", "out_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    tgts = [t for t in tgts if t in present] or ["q_proj", "v_proj"]
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=tgts))
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] r={args.lora_r} targets={tgts} trainable={ntr:,}")

    outd = Path(args.out_dir) / (args.model.replace("/", "__")
                                 + (f"__{args.variant}" if args.variant else ""))
    outd.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(outd), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, seed=args.seed,
        report_to=[], gradient_checkpointing=True, remove_unused_columns=False)

    trainer = Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    trainer.save_model(str(outd / "final"))

    json.dump({"model": args.model, "max_seq_len": args.max_seq_len,
               "seed": args.seed,
               "base_quantization": "nf4-4bit" if args.load_in_4bit else "bf16",
               "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout, "target_modules": tgts},
               "optim": {"lr": args.lr, "epochs": args.epochs,
                         "batch_size": args.batch_size, "grad_accum": args.grad_accum},
               "data": {"train": len(tr), "valid": len(va),
                        "train_truncated": tr.n_trunc, "train_dropped": tr.n_dropped,
                        "task_counts": tr.task_counts,
                        "label_counts": tr.label_counts},
               "log_history": trainer.state.log_history},
              open(outd / "run_meta.json", "w"), indent=2)
    print(f"-> {outd}")


if __name__ == "__main__":
    main()
