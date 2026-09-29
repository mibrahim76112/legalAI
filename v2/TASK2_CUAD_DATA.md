# Task 2 — CUAD clause identification

## Provenance

| | |
|---|---|
| source | `theatticusproject/cuad` :: `CUAD_v1/CUAD_v1.json` |
| version | `aok_v1.0` |
| licence | **CC BY 4.0** |
| official Atticus repository | yes — not a mirror |
| split seed | `20260919` |
| split unit | **contract** |

## Metric policy — FIXED BEFORE ANY MODEL EXISTS

| role | metric |
|---|---|
| **PRIMARY** | **F1 on the PRESENT class, reported PER CATEGORY** — this is a detection task |
| also | macro-F1 across present/absent |
| context | the constant-predictor accuracy, stated explicitly beside any accuracy figure |
| **never** | pooled binary accuracy quoted without that baseline beside it |

**Why.** Present is the minority class in 17 of 18 categories, so a constant
"absent" predictor scores high accuracy while finding nothing. It earns
**F1(present) = 0.000 by construction**, which is why F1-present is the
primary metric and pooled accuracy is not.

### Trivial baselines — constant "absent" predictor

| split | n | present | **constant-absent accuracy** |
|---|---|---|---|
| train | 6,246 | 34.5% | **65.5%** |
| dev | 1,098 | 33.3% | **66.7%** |
| test | 1,836 | 34.8% | **65.2%** |

A Task 2 accuracy of 71% is **5.5 points above a constant predictor**, not a
respectable standalone number. Quote the baseline beside it every time.

### Per-category trivial baseline — the spread is the problem

| category | present% | majority class | trivial accuracy |
|---|---|---|---|
| Anti-Assignment | 73.3% | present | **73.3%** |
| Cap On Liability | 53.9% | present | **53.9%** |
| License Grant | 50.0% | absent | **50.0%** |
| Audit Rights | 42.0% | absent | **58.0%** |
| Termination For Convenience | 35.9% | absent | **64.1%** |
| Post-Termination Services | 35.7% | absent | **64.3%** |
| Exclusivity | 35.3% | absent | **64.7%** |
| Renewal Term | 34.5% | absent | **65.5%** |
| Revenue/Profit Sharing | 32.5% | absent | **67.5%** |
| Insurance | 32.5% | absent | **67.5%** |
| Minimum Commitment | 32.4% | absent | **67.6%** |
| Non-Transferable License | 27.1% | absent | **72.9%** |
| Ip Ownership Assignment | 24.3% | absent | **75.7%** |
| Change Of Control | 23.7% | absent | **76.3%** |
| Non-Compete | 23.3% | absent | **76.7%** |
| Notice Period To Terminate Renewal | 21.8% | absent | **78.2%** |
| Uncapped Liability | 21.8% | absent | **78.2%** |
| Covenant Not To Sue | 19.6% | absent | **80.4%** |

Covenant Not To Sue has an **80.4%** trivial baseline; Anti-Assignment is the
only category where the majority class is *present*. Per-category accuracy is
therefore not comparable across categories, and pooled accuracy is dominated
by whichever categories happen to be most imbalanced.

## Selection criterion

> substantive risk-bearing provisions only; the six metadata categories (76-100% present) excluded as near-degenerate binary targets that also cannot support a risk note

**18 categories selected**, Anti-Assignment (73.3%) through
Covenant Not To Sue (19.6%).

**Excluded — the six metadata categories:** Document Name (100.0%), Parties
(99.8%), Agreement Date (92.2%), Governing Law (85.7%), Expiration Date
(81.0%), Effective Date (76.5%). Near-degenerate as binary targets, and they
cannot support a risk note. The 17 categories below 19.6% were also excluded
as too sparse.

## Output schema — mirrors ContractNLI

```json
{"present": true, "evidence": ["..."]}
```

## Dataset

| split | rows | contracts | present | absent |
|---|---|---|---|---|
| train | 6,246 | 347 | 2,155 (34.5%) | 4,091 |
| dev | 1,098 | 61 | 366 (33.3%) | 732 |
| test | 1,836 | 102 | 639 (34.8%) | 1,197 |
| **total** | **9,180** | **510** | | |

### Split is contract-level — verified

| pair | shared contracts |
|---|---|
| train ∩ dev | **0** |
| train ∩ test | **0** |
| dev ∩ test | **0** |

No contract has one clause in train and another in test.

### Evidence validation

| | |
|---|---|
| spans checked | 7,054 |
| valid | **7,054** |
| invalid | **0** |
| rows dropped | **0** |

Uses `reasoning_filters._grounded`, which wraps `text_norm.norm`. No new
normalizer and no new evidence metric were introduced.

### Examples per selected category

| category | present | absent | total |
|---|---|---|---|
| Anti-Assignment | 374 | 136 | 510 |
| Cap On Liability | 275 | 235 | 510 |
| License Grant | 255 | 255 | 510 |
| Audit Rights | 214 | 296 | 510 |
| Termination For Convenience | 183 | 327 | 510 |
| Post-Termination Services | 182 | 328 | 510 |
| Exclusivity | 180 | 330 | 510 |
| Renewal Term | 176 | 334 | 510 |
| Revenue/Profit Sharing | 166 | 344 | 510 |
| Insurance | 166 | 344 | 510 |
| Minimum Commitment | 165 | 345 | 510 |
| Non-Transferable License | 138 | 372 | 510 |
| Ip Ownership Assignment | 124 | 386 | 510 |
| Change Of Control | 121 | 389 | 510 |
| Non-Compete | 119 | 391 | 510 |
| Notice Period To Terminate Renewal | 111 | 399 | 510 |
| Uncapped Liability | 111 | 399 | 510 |
| Covenant Not To Sue | 100 | 410 | 510 |
