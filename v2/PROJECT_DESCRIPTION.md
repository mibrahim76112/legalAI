# AI Contract Review Assistant — project description

Three versions below: long form for a website, medium for a CV project entry,
and a bullet set you can trim. Every figure is from a measured run.

---

## LONG FORM (website / portfolio)

**AI Contract Review Assistant — a fine-tuned, evidence-grounded legal document
review system**

Contract review is repetitive and expensive: a reviewer must locate the
provisions that matter, check each one against their organisation's standing
requirements, and flag what needs attention. I built an end-to-end system that
does all three, and — more importantly — measured honestly whether it works.

**The system.** A single LoRA adapter fine-tuned on three tasks simultaneously,
served behind a Next.js review interface:

1. **Compliance checking** — given a contract and a playbook position, decide
   whether the contract supports it, conflicts with it, or does not address it,
   and quote the language that justifies the verdict (ContractNLI).
2. **Clause identification** — given a contract and a clause category, decide
   whether such a clause exists and return its exact text (CUAD, 18 substantive
   categories).
3. **Risk explanation** — turn a clause into one plain-English sentence a
   business reader can act on (synthetic, distilled from a Qwen3-32B teacher).

One adapter rather than three means one model to serve, version and evaluate.
Every output is anchored to quoted contract text, so a reviewer verifies rather
than trusts — a design choice supported by HCI research finding that attorneys
prefer seeking evidence over reading model explanations.

**Scale.** 22,165 training and evaluation examples over 1,117 contracts,
split at the contract level so no document appears in two splits.

**Results.** Fine-tuning produced large, consistent gains over the same base
models prompted zero-shot:

| task | metric | before | after |
|---|---|---|---|
| Compliance | macro-F1 | 0.608 | **0.875** |
| Compliance | evidence token-F1 | 0.43 | **0.81** |
| Clause ID | macro-F1 | 0.63 | **0.86** |
| Risk notes | ROUGE-L vs teacher | 0.255 | **0.450** |

Held-out **test** performance, run once and committed in advance: macro-F1
**0.852** on compliance classification, 0.734 on clause identification.

The metric that mattered commercially was Contradiction **precision** — a
reviewer handed flags that are wrong a third of the time stops trusting them.
Pre-fine-tuning it sat at 0.28-0.40 across candidate models; after, 0.62-0.67
(all figures on the full dev split).

**Engineering decisions, measured rather than assumed.**

*Sequence length.* The obvious choice, 8,192 tokens, would have destroyed the
gold evidence on 826 CUAD training rows — training the model to cite text it
cannot see. I measured evidence survival under truncation at three lengths,
dropped rather than truncated the affected rows, and profiled real memory to
show 32,768 OOMs on every candidate including a 4B model, because cost is
dominated by the 151,936-wide logits tensor rather than attention.

*Model selection.* Four models trained under identical hyperparameters and
compared by paired bootstrap on the same rows. Llama-3.1-8B won, and the result
was a reversal: it was the *worst* base model before fine-tuning. Scale bought
nothing — Qwen3-14B cost 72% more training time and scored lower.

*Quantization.* FP8 is free — indistinguishable from bf16 across 2,489
evaluation rows with 98.9% verdict agreement, at half the memory. NF4 4-bit
costs little on classification (−0.5) but **9 points of CUAD evidence quality**,
because reproducing a long verbatim span from a long document is far less
tolerant of reduced weight precision than picking one of three labels.

*Retrieval comparison.* I built a vector index over the same corpus (BM25,
dense BGE embeddings, reciprocal-rank fusion, cross-encoder reranking) and
measured recall@k. Retrieval's ceiling at k=20 (68.7% ContractNLI, 57.1% CUAD)
sits at or below what the fine-tuned model achieves reading the document
directly — and since a RAG generator can only quote what retrieval surfaces,
that ceiling bounds any pipeline built that way. Reranking proved actively
harmful when the query is a bare category label rather than a sentence.

**Evaluation rigour.** The work I am most pleased with is the measurement
discipline, because several results would have been wrong without it:

- A risk-note generation run reported a 100% filter pass rate. Inspection
  showed 85% of notes opened with the *identical six words* — the prompt
  contained a phrase the teacher was copying, and greedy decoding made it
  deterministic. Three filters were blind to it. Fixing the prompt and adding a
  templating filter took distinct openings from 6/40 to 2,049/3,044, and the
  pass rate correctly *fell* to 96.3%.
- A CUAD evidence metric was computed over positive rows only, making false
  positives structurally impossible and printing precision as exactly 1.000.
- Near-duplicate detection via MinHash found 8 cross-split document pairs above
  0.95 Jaccard that a hash-based check missed entirely; a sensitivity analysis
  showed the headline moved by less than 0.003 when they were removed.

**Deployment view.** Measured serving characteristics on one H100: 1.35 s
time-to-first-token, 14.4 ms per output token, 17 GB for weights plus KV cache,
roughly 103 contracts per hour at 17 playbook positions each.

**Stack.** PyTorch, HuggingFace Transformers and PEFT, vLLM, SLURM on an H100
cluster, sentence-transformers, Next.js and TypeScript.

---

## MEDIUM (CV project entry)

**AI Contract Review Assistant** — fine-tuned LLM system for legal document
review, with a Next.js demo interface.

Trained a single LoRA adapter on three contract-review tasks simultaneously —
compliance checking against a playbook, clause identification, and plain-English
risk explanation — over 22,165 examples from 1,117 contracts. Compared four
open-weight models (Qwen3 4B/8B/14B, Llama-3.1-8B) under identical
hyperparameters, selecting by paired bootstrap on held-out data.

Fine-tuning lifted compliance macro-F1 from 0.608 to 0.875 and evidence
extraction from 0.43 to 0.81 token-F1 on a balanced evaluation subset;
held-out test macro-F1 was 0.852.
Contradiction precision, the metric that determines whether a reviewer trusts
the flags, roughly doubled from 0.33 to 0.67.

Established the training sequence length by measuring evidence survival under
truncation (8,192 tokens would have corrupted 826 training targets) and by
profiling real GPU memory. Quantified quantization trade-offs (FP8 free, 4-bit
costs 9 points of evidence quality) and benchmarked a hybrid BM25 + dense
retrieval pipeline to show retrieval's recall ceiling sits below the fine-tuned
model's extraction.

---

## BULLETS (trim to taste)

- Fine-tuned a single LoRA adapter on **three contract-review tasks** —
  compliance verdicts, clause identification, and risk explanation — over
  **22,165 examples from 1,117 contracts**, contract-level split.
- Raised compliance macro-F1 from **0.608 to 0.875** and evidence extraction
  from **0.43 to 0.81** token-F1 over the same base model prompted zero-shot;
  held-out test **0.852**.
- Roughly doubled Contradiction **precision** (0.33 → 0.67), the metric that
  governs whether a reviewer trusts the system's flags.
- Compared **four open-weight models** under identical hyperparameters using
  **paired bootstrap significance testing**; found base-model ranking does not
  predict post-fine-tuning ranking.
- Set training sequence length by **measurement, not default** — showed 8,192
  tokens would corrupt 826 training targets, and profiled memory to prove
  32,768 is unreachable on a single 80 GB GPU.
- Quantified **quantization trade-offs**: FP8 indistinguishable from bf16
  (98.9% agreement, half the memory); NF4 4-bit costs 9 points of evidence
  quality on long documents.
- Benchmarked a **hybrid RAG pipeline** (BM25 + dense + RRF + cross-encoder
  reranking) and showed retrieval's recall ceiling bounds any RAG system below
  the fine-tuned model's direct extraction.
- Built a **Next.js review interface** where every finding is anchored to
  character offsets in the source contract, validated on 745 spans.
- Caught and corrected several **measurement artifacts** before they reached
  conclusions, including a generation failure hidden behind a 100% filter pass
  rate and a metric that made false positives structurally impossible.

**Stack:** PyTorch · HuggingFace Transformers / PEFT · vLLM · SLURM / H100 ·
sentence-transformers · Next.js / TypeScript
