#!/usr/bin/env python3
"""
PHASE 2 - LoRA SFT for document-level NDA review.

Sequences come from doc_sft_seq.build_example, which was verified token-by-token
before any training ran (verify_loss_masking.py):
  * every prompt token is labelled -100, so loss is computed on the target only.
    Targets are 2.7% of tokens; unmasked, 97.3% of the loss would be contract
    reconstruction.
  * the masked prefix IS the inference prompt byte-for-byte, so train and infer
    cannot silently diverge. gemma-4-12B-it needs this: its plain
    apply_chat_template(full_messages) render omits the empty thought channel
    that add_generation_prompt=True inserts.

Truncation at max_seq_len cuts the PROMPT from the left, never the target.
"""
import os, json, time, argparse, random, math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from doc_sft_seq import build_example, detect_think_kwarg, IGNORE_INDEX
from doc_harness import build_messages


class DocSFT(Dataset):
    def __init__(self, path, tok, think_kwarg, max_seq_len, limit=0,
                 oversample_contradiction=1, class_weight_contradiction=1.0):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if limit:
            self.rows = self.rows[:limit]
        self.tok, self.kw, self.msl = tok, think_kwarg, max_seq_len
        self.n_trunc = self.n_dropped = 0
        self.items = []
        for r in self.rows:
            ex = build_example(tok, r["messages"], think_kwarg, max_seq_len)
            if ex is None:
                self.n_dropped += 1
                continue
            self.n_trunc += ex["truncated"]
            ex["verdict"] = r["verdict"]
            # Per-example loss weight. Gemma's baseline SFT traded Contradiction
            # recall (0.716 -> 0.547) for precision; upweighting the minority
            # class is the loss-side lever on that trade.
            ex["weight"] = (class_weight_contradiction
                            if r["verdict"] == "Contradiction" else 1.0)
            self.items.append(ex)

        # Data-side lever on the same trade: duplicate Contradiction rows so the
        # class is seen more often. 841 of 7188 train rows are Contradiction.
        self.n_oversampled = 0
        if oversample_contradiction > 1:
            extra = [e for e in self.items if e["verdict"] == "Contradiction"]
            for _ in range(oversample_contradiction - 1):
                self.items.extend(extra)
                self.n_oversampled += len(extra)
        self.class_counts = dict(Counter(e["verdict"] for e in self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        lab.append(b["labels"] + [IGNORE_INDEX] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att),
            "example_weight": torch.tensor([b.get("weight", 1.0) for b in batch],
                                           dtype=torch.float)}


def load_model(model_id, dtype=torch.bfloat16):
    """gemma-4-12B-it is Gemma4UnifiedForConditionalGeneration, not a plain
    CausalLM, so try the conditional-generation classes before giving up."""
    from transformers import AutoModelForCausalLM, AutoConfig
    last = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModelForVision2Seq", "AutoModelForSeq2SeqLM"):
        try:
            import transformers
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            m = cls.from_pretrained(model_id, dtype=dtype,
                                    trust_remote_code=True, device_map=None)
            print(f"[load] {cls_name} -> {type(m).__name__}")
            return m
        except Exception as e:
            last = f"{cls_name}: {type(e).__name__}: {str(e)[:120]}"
    raise SystemExit(f"could not load {model_id}. last: {last}")


class WeightedTrainer(__import__("transformers").Trainer):
    """Scales each example's mean token loss by its class weight.

    Implemented at example level rather than token level because the verdict is
    only a few tokens of a target that is mostly JSON scaffolding; weighting the
    whole example is what actually shifts the class balance the model sees.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        w = inputs.pop("example_weight", None)
        labels = inputs["labels"]
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = out.logits[:, :-1, :]
        tgt = labels[:, 1:]
        mask = tgt.ne(IGNORE_INDEX)
        lp = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1).clamp_min(0), reduction="none").view(tgt.shape)
        lp = lp * mask
        per_ex = lp.sum(1) / mask.sum(1).clamp_min(1)
        if w is not None:
            w = w.to(per_ex.device)
            loss = (per_ex * w).sum() / w.sum().clamp_min(1e-6)
        else:
            loss = per_ex.mean()
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="doc_sft/train.jsonl")
    ap.add_argument("--valid", default="doc_sft/valid.jsonl")
    ap.add_argument("--out-dir", default=os.environ.get("SFT_DIR", "sft_out"))
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-n", type=int, default=256,
                    help="dev subset for loss; full dev eval happens in Phase 3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oversample-contradiction", type=int, default=1,
                    help="duplicate Contradiction train rows N times (1 = off)")
    ap.add_argument("--class-weight-contradiction", type=float, default=1.0,
                    help="loss multiplier on Contradiction examples (1.0 = off)")
    ap.add_argument("--variant", default="", help="suffix for the output dir")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="set high to keep a checkpoint sweep for metric-vs-loss analysis")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    from peft import LoraConfig, get_peft_model
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw, default_think = detect_think_kwarg(tok, build_messages("x", "y"))
    print(f"[template] think kwarg={kw} default_is_thinking={default_think} "
          f"-> training with thinking DISABLED (matches Phase 1/3 inference)")

    tr = DocSFT(args.train, tok, kw, args.max_seq_len, args.limit,
                args.oversample_contradiction, args.class_weight_contradiction)
    va = DocSFT(args.valid, tok, kw, args.max_seq_len, args.eval_n)
    print(f"[data] train={len(tr)} (truncated {tr.n_trunc}, dropped {tr.n_dropped}, "
          f"oversampled +{tr.n_oversampled}) valid={len(va)}")
    print(f"[data] train class balance: {tr.class_counts}")
    if args.class_weight_contradiction != 1.0:
        print(f"[data] Contradiction loss weight = {args.class_weight_contradiction}")

    model = load_model(args.model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in targets if t in present] or ["q_proj", "v_proj"]
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] r={args.lora_r} alpha={args.lora_alpha} targets={targets}")
    print(f"[lora] trainable {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    outd = Path(args.out_dir) / (args.model.replace("/", "__")
                                 + (f"__{args.variant}" if args.variant else ""))
    outd.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(outd), seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[], gradient_checkpointing=True, remove_unused_columns=False,
    )
    use_w = (args.class_weight_contradiction != 1.0)
    TrainerCls = WeightedTrainer if use_w else Trainer
    trainer = TrainerCls(model=model, args=targs, train_dataset=tr,
                         eval_dataset=va,
                         data_collator=lambda b: collate(b, tok.pad_token_id))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    ev = trainer.evaluate()
    model.save_pretrained(str(outd / "adapter"))
    tok.save_pretrained(str(outd / "adapter"))

    hist = [h for h in trainer.state.log_history]
    manifest = {
        "model": args.model, "seed": args.seed,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout, "target_modules": targets,
                 "trainable_params": trainable, "total_params": total},
        "optim": {"lr": args.lr, "epochs": args.epochs,
                  "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                  "effective_batch": args.batch_size * args.grad_accum,
                  "scheduler": "cosine", "warmup_ratio": 0.03},
        "variant": args.variant or "baseline",
        "oversample_contradiction": args.oversample_contradiction,
        "class_weight_contradiction": args.class_weight_contradiction,
        "train_class_counts": tr.class_counts,
        "data": {"train": len(tr), "valid": len(va),
                 "oversampled": tr.n_oversampled,
                 "truncated": tr.n_trunc, "dropped": tr.n_dropped,
                 "max_seq_len": args.max_seq_len},
        "think_kwarg": kw, "default_is_thinking": default_think,
        "final_train_loss": res.training_loss,
        "final_eval_loss": ev.get("eval_loss"),
        "wall_seconds": round(wall, 1), "peak_gpu_gb": round(peak, 2),
        "log_history": hist,
    }
    (outd / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Emit curves with the manifest so every run is inspectable without a
    # separate step: convergence, train-vs-eval gap, grad-norm stability and
    # late-run loss variance are what decide rank / LR / batch changes.
    try:
        import subprocess, sys as _s
        subprocess.run([_s.executable, "plot_training.py", "--sft-dir",
                        str(Path(args.out_dir))], check=False)
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print(f"\n[done] train_loss={res.training_loss:.4f} "
          f"eval_loss={ev.get('eval_loss'):.4f} wall={wall/60:.1f}min "
          f"peak={peak:.1f}GB -> {outd}")


if __name__ == "__main__":
    main()
