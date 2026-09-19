# Context-length budget for document-level SFT

Task moves from clause-level to document-level: input is a full NDA plus one
policy question, output is `{"verdict": ..., "evidence": [...]}`.

All counts are **measured end-to-end** on the real training sequences
(chat template + system prompt + full document + question, and the
serialized target JSON). Nothing is estimated by adding part counts, because
subword boundaries make token counts non-additive.

Corpus: **607 documents** (train 423 / dev 61 / test 123), **10319 (document, hypothesis) pairs**.


## Part 1 — Document length distribution

Tokens for `document.text` alone, all 607 documents.

| tokenizer | min | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| Qwen3-14B | 275 | 1915 | 4039 | 4726 | 6378 | 11532 |
| Qwen3-8B | 275 | 1915 | 4039 | 4726 | 6378 | 11532 |
| granite-4.2-8b | 282 | 1939 | 4068 | 4711 | 6325 | 11589 |
| phi-4 | 272 | 1901 | 4009 | 4632 | 6243 | 11340 |

Documents exceeding each threshold (of 607):

| tokenizer | >4096 | >8192 | >16384 | >32768 |
|---|---|---|---|---|
| Qwen3-14B | 57 | 2 | 0 | 0 |
| Qwen3-8B | 57 | 2 | 0 | 0 |
| granite-4.2-8b | 58 | 2 | 0 | 0 |
| phi-4 | 56 | 1 | 0 | 0 |

### Documents over 16384 tokens

None, under any of the four tokenizers.


## Part 2 — Full sequence budget (measured)

### Output (target JSON) tokens, by verdict

| tokenizer | verdict | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| Qwen3-14B | Entailment | 5017 | 86 | 230 | 325 | 520 |
| Qwen3-14B | Contradiction | 1156 | 96 | 257 | 384 | 601 |
| Qwen3-14B | NotMentioned | 4146 | 16 | 16 | 16 | 16 |
| Qwen3-8B | Entailment | 5017 | 86 | 230 | 325 | 520 |
| Qwen3-8B | Contradiction | 1156 | 96 | 257 | 384 | 601 |
| Qwen3-8B | NotMentioned | 4146 | 16 | 16 | 16 | 16 |
| granite-4.2-8b | Entailment | 5017 | 86 | 230 | 326 | 533 |
| granite-4.2-8b | Contradiction | 1156 | 97 | 259 | 387 | 605 |
| granite-4.2-8b | NotMentioned | 4146 | 16 | 16 | 16 | 16 |
| phi-4 | Entailment | 5017 | 85 | 229 | 324 | 520 |
| phi-4 | Contradiction | 1156 | 96 | 257 | 384 | 601 |
| phi-4 | NotMentioned | 4146 | 16 | 16 | 16 | 16 |

### Total sequence length (input + output)

| tokenizer | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| Qwen3-14B | 2108 | 4247 | 4925 | 6710 | 11882 |
| Qwen3-8B | 2108 | 4247 | 4925 | 6710 | 11882 |
| granite-4.2-8b | 2138 | 4288 | 4974 | 6743 | 11944 |
| phi-4 | 2093 | 4213 | 4867 | 6622 | 11688 |

Sequences exceeding each threshold:

| tokenizer | total pairs | >4096 | >8192 | >16384 | >32768 |
|---|---|---|---|---|---|
| Qwen3-14B | 10319 | 1139 (11.0%) | 39 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| Qwen3-8B | 10319 | 1139 (11.0%) | 39 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| granite-4.2-8b | 10319 | 1177 (11.4%) | 49 (0.5%) | 0 (0.0%) | 0 (0.0%) |
| phi-4 | 10319 | 1100 (10.7%) | 36 (0.3%) | 0 (0.0%) | 0 (0.0%) |

## Part 3 — Tokenizer efficiency

Mean tokens per document. Baseline = **phi-4** (most efficient).

| tokenizer | mean tokens/doc | vs best | chars/token |
|---|---|---|---|
| phi-4 | 2146 | baseline | 5.21 |
| Qwen3-14B | 2164 | +0.8% | 5.17 |
| Qwen3-8B | 2164 | +0.8% | 5.17 |
| granite-4.2-8b | 2193 | +2.2% | 5.10 |

Spread across the four tokenizers is **under 2.2%**. For a 2,000-token contract
that is ~47 tokens, well inside the noise of a single review. **Tokenizer
efficiency is not a usable model-selection criterion on this corpus** - the
per-review cost difference it implies is negligible next to the parameter-count
difference (8B vs 14B) that actually drives serving cost. Reporting it as a
differentiator would overstate a 2% effect.

## Part 4 — Recommendation

### `max_seq_len = 8192`

| criterion | value |
|---|---|
| p99 of total sequence length (worst tokenizer, granite) | 6,743 |
| chosen power of two covering p99 | **8192** |
| headroom over p99 | ~21% |
| max observed total sequence | 11,944 |
| sequences over 8192 | 36-49 of 10,319 (**0.3-0.5%**) |
| documents involved | 3-4 of 607 |
| **pairs where truncation destroys gold evidence** | **3** (2 for phi-4) |

### What gets truncated, and what must be dropped instead

Of the ~39 pairs over budget, most truncate harmlessly: the cut falls after all
gold spans, so the target still cites visible text. **Three do not**, and these
must be **dropped from training, not truncated** - a target citing text the
model cannot see teaches it to fabricate citations, which is a corrupted label
rather than a shorter example.

| doc_id | file_name | split | hypothesis | verdict | spans lost | severity |
|---|---|---|---|---|---|---|
| 622 | `1628908_0001193125-15-169530_d838828dex1016.htm` | train | nda-10 | Entailment | **1 of 1** | total - only evidence ends at char 53,168, cut lands ~37,800 |
| 622 | `1628908_0001193125-15-169530_d838828dex1016.htm` | train | nda-19 | Entailment | 1 of 2 | partial |
| 162 | `Mutual%20confidentiality%20and%20NDA%20-%20Final%20(2)%20(6)%2003%20July%202015.pdf` | train | nda-19 | Entailment | 1 of 4 | partial (fits under phi-4, so phi-4 loses only 2 pairs) |

All three are in **train**, so dev and test are untouched and no reported metric
is affected. Dropping them costs **0.03%** of the training pairs.

The two source documents are the corpus extremes: doc 622 is the longest at
11,340-11,589 tokens, doc 162 the second longest at 8,265-8,280.

### Why not 4096 or 16384

- **4096** would put ~11% of pairs (1,100-1,177) over budget. That is far too
  much truncation to audit span-by-span, and it cuts into the body of the
  distribution rather than its tail.
- **16384** is the zero-loss option: **no sequence exceeds it under any of the
  four tokenizers**. But it doubles activation memory and attention cost to
  rescue 3 pairs out of 10,319. If training memory turns out not to bind, take
  16384 and drop nothing; otherwise 8192 plus the three exclusions is the better
  trade.

### Note on the NotMentioned target

Measured with `"evidence": []`, which is a fixed **16 tokens** for every
NotMentioned row - 40% of the corpus, and the reason the output distribution is
so cheap overall.

The system prompt as written instructs the model to return `'no related clause'`
when nothing applies, which does not match an empty list. Using that string
instead costs **19 tokens** (+3). The budget is unaffected either way, but
**prompt and target must agree** - training on `[]` while the prompt asks for
`'no related clause'` teaches the model to ignore an explicit instruction.
Pick one before building the SFT data.

### Summary

| | |
|---|---|
| **max_seq_len** | **8192** |
| Pairs dropped | 3 (doc 622 x2, doc 162 x1 - all train, all Entailment) |
| Pairs safely truncated | ~36 |
| Pairs untouched | 10,280 (99.6%) |
| Fallback | 16384 if memory allows - zero loss, nothing dropped |
