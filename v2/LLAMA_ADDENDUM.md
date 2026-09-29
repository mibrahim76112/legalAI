# Addendum — Llama-3.1-8B-Instruct added to the pre-SFT screen

## Why this run exists

The original screen contained no Llama model. The user asked for one, since a
model-selection argument that never tested the most widely deployed open
8B-class instruct model is weak in exactly the place a reviewer will press.

`meta-llama/Meta-Llama-3.1-8B-Instruct` is gated on the Hub and the compute
nodes have no interactive path to accept a licence. The weights were taken from
the `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror. **This is a mirror, not a
derivative**, and that claim was verified rather than assumed:

- `config.json` matches the Meta release — `LlamaForCausalLM`, 32 layers,
  hidden 4096, 8 KV heads, `rope_type: llama3` with factor 8 over an original
  8,192 window, vocab 128,256.
- `max_position_embeddings` 131,072, matching the 3.1 release.
- No adapter, merge, or quantization artifacts in the snapshot.

If the presentation names this model, it should name the mirror too. Reporting
it as "Llama-3.1-8B" without qualification would be a small misstatement.

## Conditions

Identical to every other cell in the screen: full NDA as context, one policy
question per call, zero-shot, no thinking, `max_new_tokens = 1024`, greedy,
same prompt, same parser, same normalizer, dev split, n = 1,037. Nothing was
tuned for Llama that was not tuned for the others.

## Result

| model | macro-F1 | 95% CI | accuracy |
|---|---|---|---|
| gemma-4-12B-it | **0.732** | [0.700, 0.763] | 0.792 |
| Qwen3-14B | 0.701 | [0.668, 0.733] | 0.753 |
| granite-4.2-8b | 0.623 | [0.590, 0.654] | 0.675 |
| Qwen3-4B | 0.615 | [0.579, 0.650] | 0.669 |
| **Meta-Llama-3.1-8B-Instruct** | **0.592** | **[0.558, 0.626]** | 0.657 |
| Qwen3-8B | 0.553 | [0.519, 0.585] | 0.623 |

Llama places fifth of six. Read the intervals before the ranking: Llama's CI
overlaps Qwen3-4B's and granite's substantially, so **Llama is not separated
from the middle of the field**. It *is* separated from gemma-4-12B-it, whose
interval does not overlap it.

The conclusion is that adding Llama did not change the selection. gemma and
Qwen3-14B remain the only models separated from the pack, and they were already
the leaders.

## The interesting part — Llama fails differently

Ranking Llama fifth undersells what the run actually showed. Its error profile
is the **opposite** of Qwen's, and that is more informative than its position.

| model | Contradictions flagged | real | precision | recall | over-flag |
|---|---|---|---|---|---|
| gemma-4-12B-it | 147 | 68 | 0.46 | 0.72 | 1.5x |
| Qwen3-14B | 193 | 77 | 0.40 | 0.81 | 2.0x |
| Qwen3-4B | 175 | 61 | 0.35 | 0.64 | 1.8x |
| **Meta-Llama-3.1-8B** | **146** | **48** | **0.33** | **0.51** | **1.5x** |
| granite-4.2-8b | 234 | 76 | 0.32 | 0.80 | 2.5x |
| Qwen3-8B | 284 | 80 | 0.28 | 0.84 | 3.0x |

Dev carries 95 gold Contradictions. Every Qwen model over-flags — Qwen3-8B
raises 284 flags for 95 real conflicts, a 3x cry-wolf rate. Llama raises 146,
the joint-lowest volume, but **catches only half the real conflicts**
(recall 0.505, the lowest in the field by a wide margin; the next lowest is
0.642).

So Llama is not a quieter version of Qwen. It is a differently-broken model:
where the Qwen family escalates everything and buries the reviewer, Llama stays
quiet and **misses half the conflicts that matter**. For a compliance review
tool those are not interchangeable failures. Over-flagging wastes reviewer time;
under-flagging lets a conflicting NDA through. Llama's profile is the more
dangerous of the two even though its flag volume looks healthier.

This shows up again in evidence behaviour:

| model | strict evidence F1 | false-abstention rate |
|---|---|---|
| gemma-4-12B-it | 0.432 | 0.057 |
| Qwen3-8B | 0.296 | 0.013 |
| Qwen3-4B | 0.356 | 0.122 |
| granite-4.2-8b | 0.525 | 0.134 |
| **Meta-Llama-3.1-8B** | 0.393 | **0.342** |

Llama returns **no evidence at all on 34.2% of rows where gold evidence
exists** — two and a half times the next-worst model. Its middling evidence F1
is therefore produced by a small number of confident citations plus a large
number of silences, not by consistent extraction.

## What this changes

Nothing about model selection, and that is a legitimate result worth reporting.
The value of the run is that the "why no Llama?" question now has a measured
answer instead of an omission, and the answer carries a finding of its own:
**under-flagging and over-flagging are both present in the candidate pool, and
they are model-family traits, not a single shared weakness that fine-tuning can
be assumed to fix.** A recall-side fix aimed at Llama and a precision-side fix
aimed at Qwen are different interventions.

Source predictions: `doc_results/raw__NousResearch__Meta-Llama-3.1-8B-Instruct__nothink__dev__mnt1024.jsonl`
