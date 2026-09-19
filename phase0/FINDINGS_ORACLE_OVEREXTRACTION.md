# Oracle evidence + over-extraction breakdown (dev split)

## CUAD — hypothesis rejected by measurement

Not run, and will not be run, including as a negative control. The decision rule
required fabricated + missed >= 40% of evidence failures. Measured 14.3% / 16.3%
/ 14.6% across Qwen3-4B / 8B / 14B. Replication across three models is what makes
the SKIP robust. Logged as a hypothesis rejected by measurement, not deferred.

## Oracle evidence: perfect retrieval does NOT improve verdicts

614 gold-bearing dev rows (519 Entailment, 95 Contradiction). NotMentioned rows
excluded because their gold evidence is empty, so supplying excerpts would
announce the label. This is therefore an upper bound on E-vs-C discrimination
given perfect retrieval, not a three-way oracle.

| model | normal acc | oracle acc | gain |
|---|---|---|---|
| Qwen3-4B | 0.912 | 0.904 | **−0.008** |
| Qwen3-8B | 0.907 | 0.910 | **+0.003** |
| Qwen3-14B | 0.912 | 0.909 | **−0.003** |

**A clean null.** Handing the model the exact spans a senior reviewer identified
changes verdict accuracy by less than one point in either direction, on all
three models. Contradiction P/R is likewise flat.

### Why this matters

The observational correlation is strong: verdict accuracy is 97.7% when evidence
is fully recovered and 76.3% when it is not, a 21-point gap. That gap is
**confounded, not causal.** The oracle run breaks the confound: supplying correct
evidence does not transfer the 21 points. Both are downstream of the same thing —
rows where the governing language is ambiguous are hard for retrieval AND for
judgment, and fixing one does not fix the other.

**Consequence: improving evidence extraction will not improve verdicts.** They
must be treated as separate objectives. This also independently supports the
CUAD SKIP: even if CUAD had improved extraction, the verdict metrics would not
have moved.

## Over-extraction (category b): PARTIALLY shared root cause

Rate = over-extractions / NotMentioned rows, per hypothesis, mean of three models.

| hyp | NM rows | 4B | 8B | 14B | mean | in partial-overlap suspect set |
|---|---|---|---|---|---|---|
| nda-7 | 8 | 87.5% | 87.5% | 62.5% | **79.2%** | **yes** |
| nda-20 | 36 | 72.2% | 75.0% | 75.0% | **74.1%** | **yes** |
| nda-5 | 5 | 60.0% | 40.0% | 40.0% | 46.7% | no |
| nda-4 | 8 | 37.5% | 25.0% | 37.5% | 33.3% | no |
| nda-16 | 37 | 21.6% | 18.9% | 10.8% | 17.1% | yes |
| nda-18 | 50 | 2.0% | 2.0% | 4.0% | **2.7%** | yes |
| nda-11 | 53 | 0.0% | 0.0% | 0.0% | **0.0%** | no |

**Answer: partially, and the partial answer is the informative one.**

Two of the four partial-scope suspects (nda-7, nda-20) are also the two worst
over-extractors by a wide margin — 74-79% against a corpus median near 13%. For
those two, **one root cause explains three failure modes**: partial overlap,
Contradiction false positives, and over-extraction.

But the mapping is not clean. **nda-18 over-extracts at 2.7%**, near the floor,
despite being in the partial-overlap suspect set. And nda-5 and nda-4 are heavy
over-extractors without being partial-overlap suspects.

Reading the positions explains the split. The two that fail everywhere —
nda-7 "may share **some** CI with **some** third-parties" and nda-20 "may retain
**some** CI **even after** return or destruction" — are **permissive** positions
with nested quantifiers. Almost any disclosure or retention clause looks
partially relevant, so the model finds something to cite whether or not the
contract addresses the specific permission. nda-18, "shall not solicit some of
DP's representatives", is **prohibitive and narrow**: non-solicitation language
is either present or absent, so abstention is easy even though span boundaries
are contested when it is present.

So the shared root cause is **permissive, nested-quantifier positions**, not
partial-scope wording in general. nda-5 and nda-4 fit that description too
("may share some", "shall not use **any**"), which is why they over-extract
without being span-boundary problems.

**This is an independent abstention failure for some hypotheses and a shared
cause for others**, and the distinguishing feature is permissive vs prohibitive
framing. Worth naming separately in the writeup rather than folding into one
story.

## Caveats

- Dev split throughout. Test untouched.
- nda-7 and nda-5 have very few NotMentioned rows (8 and 5 per model), so their
  rates are high-variance. nda-20 (36 rows) and nda-16 (37) are better supported.
- Oracle run is E-vs-C only by construction; it says nothing about NotMentioned
  abstention.
