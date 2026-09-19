# Phase 2, model 1 of 3 — gemma-4-12B-it LoRA SFT

Dev split (1037 pairs), same harness and settings as the base run. Base
comparison uses the **1024-token re-run**, so the cap is identical on both sides
and the delta is attributable to fine-tuning alone.

## Training configuration

| | |
|---|---|
| LoRA | r=16, alpha=32, dropout 0.05 |
| target modules | q,k,v,o,gate,up,down_proj |
| trainable | 65,568,768 / 12,025,298,944 (**0.545%**) |
| LR / schedule | 1e-4, cosine, warmup 0.03 |
| epochs | 1.0 |
| batch / grad-accum | 1 x 16 (effective 16) |
| seed | 42 |
| max_seq_len | 8192 |
| train / valid | 7188 / 256 (31 truncated, 0 dropped) |
| wall clock | 177.1 min |
| peak GPU | 58.6 GB |
| final train / eval loss | 0.0497 (run mean) / 0.0292 |

## Loss curves

| step | train | | step | eval |
|---|---|---|---|---|
| 10 | 1.0888 | | 50 | 0.0286 |
| 50 | 0.0281 | | 250 | 0.0406 |
| 250 | 0.0165 | | 450 | 0.0292 |
| 450 | 0.0187 | | | |

**Read the loss with suspicion.** It reaches its floor within ~50 of 450 steps
and then flatlines; eval loss at step 450 (0.0292) is no better than at step 50
(0.0286). Most target tokens are JSON scaffolding, which fits almost
immediately, so loss mostly measures format acquisition. **The dev eval, not the
loss, is what shows the fine-tune worked** — and it did, substantially.

## Results vs base (paired bootstrap, 1200 resamples)

| metric | base | SFT | delta | 95% CI | |
|---|---|---|---|---|---|
| macro-F1 | 0.732 | **0.789** | +0.057 | [+0.020, +0.097] | **separated** |
| **Contradiction precision** | 0.463 | **0.693** | **+0.231** | [+0.138, +0.326] | **separated** |
| Contradiction recall | 0.716 | 0.547 | **-0.170** | [-0.269, -0.069] | **separated** |
| Contradiction F1 | 0.562 | 0.612 | +0.049 | [-0.036, +0.132] | within noise |
| **evidence F1 strict** | 0.432 | **0.693** | **+0.262** | [+0.220, +0.305] | **separated** |
| **joint strict** | 0.464 | **0.699** | **+0.236** | [+0.204, +0.268] | **separated** |
| NotMentioned recall | 0.697 | 0.915 | +0.217 | | |
| accuracy | 0.792 | 0.857 | +0.066 | | |

## What moved

**The veto metric moved, by a lot.** Contradiction precision 0.463 -> 0.693.
The cry-wolf rate falls from **54% to 31%**, and the model raises 75 flags where
the base raised 147 against 95 real conflicts — over-flagging of 1.5x drops to
under-flagging of 0.8x.

**Evidence extraction was the capability bottleneck and it moved most**:
strict F1 0.432 -> 0.693, +0.262. This is the single largest gain and it
validates evidence spans as the strongest SFT justification.

**Joint accuracy 0.464 -> 0.699.** Fully-correct predictions go from under half
to about seven in ten. That is the product number.

**Abstention improved sharply**: NotMentioned recall 0.697 -> 0.915.

## The trade-off that must be reported

**Contradiction recall fell 0.716 -> 0.547, and the drop is statistically
separated.** The fine-tuned model now misses **45% of real conflicts**, against
28% for the base.

Contradiction F1 is +0.049 but **within noise** — so on a symmetric view the
class did not clearly improve. What changed is *which* error it makes: the base
over-flags, the fine-tune under-flags.

Which is better depends on a cost model we have not been given:

- If a missed conflict is catastrophic and reviewer time is cheap, the base's
  error profile is safer and this fine-tune is a regression on that axis.
- If reviewer trust is the binding constraint — the original argument for
  switching the veto metric to precision — the fine-tune is the clear win.

**This is tunable, not inherent.** Class weighting, a decision threshold on the
Contradiction logit, or oversampling the 841 Contradiction training rows should
recover recall at some precision cost. Worth one experiment before Phase 3.

## Verdict

The fine-tune succeeded on every axis it was meant to: precision, evidence,
joint, abstention. It is not a free win — Contradiction recall paid for the
precision gain — and that trade needs a decision from the cost side before it
is presented as an unqualified success.
