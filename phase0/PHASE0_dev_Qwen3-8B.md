# Phase 0 diagnostic — dev_Qwen3-8B split

Predictions: `raw____localscratch__ibi761.21892914.0__merged_Qwen_Qwen3-8B__nothink__dev__sft.jsonl`  (1037 rows)

Analysis only. **Dev split** — test is reserved and untouched.


## 0.1 Per-hypothesis breakdown

| hyp | n | gold E/C/NM | macro-F1 | acc | ev F1 | ev exact | ev partial |
|---|---|---|---|---|---|---|---|
| nda-20 | 61 | 13/12/36 | 0.571 | 0.525 | **0.448** | 0.393 | 0.767 |
| nda-16 | 61 | 22/2/37 | 0.540 | 0.803 | **0.591** | 0.705 | 0.671 |
| nda-18 | 61 | 11/0/50 | 0.609 | 0.951 | **0.667** | 0.902 | 0.674 |
| nda-19 | 61 | 43/0/18 | 0.550 | 0.853 | **0.685** | 0.623 | 0.692 |
| nda-7 | 61 | 39/14/8 | 0.608 | 0.787 | **0.696** | 0.541 | 0.755 |
| nda-17 | 61 | 9/8/44 | 0.767 | 0.836 | **0.703** | 0.820 | 0.824 |
| nda-10 | 61 | 29/0/32 | 0.601 | 0.902 | **0.720** | 0.770 | 0.732 |
| nda-4 | 61 | 50/3/8 | 0.712 | 0.885 | **0.721** | 0.607 | 0.711 |
| nda-1 | 61 | 19/10/32 | 0.781 | 0.836 | **0.723** | 0.787 | 0.660 |
| nda-2 | 61 | 6/44/11 | 0.540 | 0.836 | **0.738** | 0.639 | 0.744 |
| nda-15 | 61 | 39/0/22 | 0.581 | 0.885 | **0.743** | 0.705 | 0.780 |
| nda-3 | 61 | 45/0/16 | 0.591 | 0.902 | **0.769** | 0.705 | 0.744 |
| nda-5 | 61 | 54/2/5 | 0.524 | 0.918 | **0.779** | 0.656 | 0.784 |
| nda-11 | 61 | 8/0/53 | 0.613 | 0.967 | **0.857** | 0.967 | 0.750 |
| nda-13 | 61 | 47/0/14 | 0.651 | 0.984 | **0.857** | 0.803 | 0.893 |
| nda-12 | 61 | 40/0/21 | 0.667 | 1.000 | **0.889** | 0.869 | 0.899 |
| nda-8 | 61 | 45/0/16 | 0.638 | 0.967 | **0.918** | 0.885 | 0.894 |

## 0.3 Evidence-failure attribution

282 failures of 1037 rows (27.2%).

| category | n | % of failures | % of all rows |
|---|---|---|---|
| (a) missed — pred empty, gold non-empty | 36 | 12.8% | 3.5% |
| (b) over-extraction — pred non-empty, gold empty | 75 | 26.6% | 7.2% |
| (c) fabricated — span absent from contract | 10 | 3.5% | 1.0% |
| (d) partial — grounded, partial overlap with gold | 127 | 45.0% | 12.2% |
| (e) wrong span — grounded, no overlap | 34 | 12.1% | 3.3% |

## 0.3b How near are the near-misses? (item 8)

Gold spans recovered / required, over 207 gold-bearing failures:

| p10 | p25 | p50 | p75 | p90 | share ≥0.5 | share =0 |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.40 | 0.50 | 0.67 | 48.8% | 35.7% |

Strict evidence F1 (**primary, unchanged**): **0.7427**

Partial-credit variant (reported alongside, NOT a substitute): 0.7715


## 0.4 Positional check

Gold-evidence first-half share, per hypothesis (min 30.2%, max 96.0%):

| hyp | first-half share | n spans | ev F1 |
|---|---|---|---|
| nda-2 | 96.0% | 101 | 0.738 |
| nda-1 | 94.8% | 58 | 0.723 |
| nda-7 | 94.2% | 104 | 0.696 |
| nda-5 | 93.1% | 102 | 0.779 |
| nda-3 | 90.7% | 75 | 0.769 |
| nda-4 | 90.0% | 90 | 0.721 |
| nda-8 | 84.6% | 78 | 0.918 |
| nda-10 | 83.6% | 67 | 0.720 |
| nda-12 | 74.3% | 101 | 0.889 |
| nda-13 | 72.7% | 121 | 0.857 |
| nda-17 | 69.6% | 23 | 0.703 |
| nda-11 | 66.7% | 9 | 0.857 |
| nda-18 | 56.7% | 30 | 0.667 |
| nda-16 | 43.1% | 51 | 0.591 |
| nda-15 | 35.3% | 68 | 0.743 |
| nda-19 | 30.2% | 86 | 0.685 |
| nda-20 | 30.2% | 63 | 0.448 |

## 0.5 Decision rule

- worst quartile by evidence F1 (4 hypotheses): ['nda-16', 'nda-18', 'nda-19', 'nda-20']
- (c) fabricated + (a) missed = **16.3%** of all failures  (threshold ≥40% → PROCEED)
- (d) partial + (e) wrong span **within the worst quartile** = **11.0%** of all failures

### BRANCH TAKEN: **SKIP Phases 1 and 2**

