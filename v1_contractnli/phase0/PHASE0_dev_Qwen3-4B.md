# Phase 0 diagnostic — dev_Qwen3-4B split

Predictions: `raw____localscratch__ibi761.21892925.0__merged_Qwen_Qwen3-4B__nothink__dev__sft.jsonl`  (1037 rows)

Analysis only. **Dev split** — test is reserved and untouched.


## 0.1 Per-hypothesis breakdown

| hyp | n | gold E/C/NM | macro-F1 | acc | ev F1 | ev exact | ev partial |
|---|---|---|---|---|---|---|---|
| nda-20 | 61 | 13/12/36 | 0.600 | 0.557 | **0.455** | 0.410 | 0.810 |
| nda-16 | 61 | 22/2/37 | 0.529 | 0.787 | **0.638** | 0.721 | 0.719 |
| nda-10 | 61 | 29/0/32 | 0.579 | 0.869 | **0.667** | 0.721 | 0.697 |
| nda-7 | 61 | 39/14/8 | 0.593 | 0.770 | **0.667** | 0.508 | 0.709 |
| nda-19 | 61 | 43/0/18 | 0.573 | 0.885 | **0.676** | 0.623 | 0.674 |
| nda-1 | 61 | 19/10/32 | 0.794 | 0.853 | **0.711** | 0.787 | 0.625 |
| nda-2 | 61 | 6/44/11 | 0.653 | 0.853 | **0.716** | 0.623 | 0.680 |
| nda-4 | 61 | 50/3/8 | 0.719 | 0.902 | **0.727** | 0.607 | 0.737 |
| nda-18 | 61 | 11/0/50 | 0.609 | 0.951 | **0.737** | 0.918 | 0.705 |
| nda-3 | 61 | 45/0/16 | 0.595 | 0.918 | **0.753** | 0.689 | 0.752 |
| nda-5 | 61 | 54/2/5 | 0.469 | 0.902 | **0.783** | 0.656 | 0.782 |
| nda-17 | 61 | 9/8/44 | 0.799 | 0.885 | **0.788** | 0.885 | 0.794 |
| nda-15 | 61 | 39/0/22 | 0.594 | 0.902 | **0.806** | 0.770 | 0.810 |
| nda-13 | 61 | 47/0/14 | 0.636 | 0.967 | **0.815** | 0.754 | 0.850 |
| nda-11 | 61 | 8/0/53 | 0.613 | 0.967 | **0.857** | 0.967 | 0.750 |
| nda-12 | 61 | 40/0/21 | 0.654 | 0.984 | **0.907** | 0.885 | 0.941 |
| nda-8 | 61 | 45/0/16 | 0.652 | 0.984 | **0.930** | 0.902 | 0.916 |

## 0.3 Evidence-failure attribution

279 failures of 1037 rows (26.9%).

| category | n | % of failures | % of all rows |
|---|---|---|---|
| (a) missed — pred empty, gold non-empty | 38 | 13.6% | 3.7% |
| (b) over-extraction — pred non-empty, gold empty | 72 | 25.8% | 6.9% |
| (c) fabricated — span absent from contract | 2 | 0.7% | 0.2% |
| (d) partial — grounded, partial overlap with gold | 126 | 45.2% | 12.2% |
| (e) wrong span — grounded, no overlap | 41 | 14.7% | 4.0% |

## 0.3b How near are the near-misses? (item 8)

Gold spans recovered / required, over 207 gold-bearing failures:

| p10 | p25 | p50 | p75 | p90 | share ≥0.5 | share =0 |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.33 | 0.50 | 0.67 | 46.4% | 39.1% |

Strict evidence F1 (**primary, unchanged**): **0.7447**

Partial-credit variant (reported alongside, NOT a substitute): 0.7668


## 0.4 Positional check

Gold-evidence first-half share, per hypothesis (min 30.2%, max 96.0%):

| hyp | first-half share | n spans | ev F1 |
|---|---|---|---|
| nda-2 | 96.0% | 101 | 0.716 |
| nda-1 | 94.8% | 58 | 0.711 |
| nda-7 | 94.2% | 104 | 0.667 |
| nda-5 | 93.1% | 102 | 0.783 |
| nda-3 | 90.7% | 75 | 0.753 |
| nda-4 | 90.0% | 90 | 0.727 |
| nda-8 | 84.6% | 78 | 0.930 |
| nda-10 | 83.6% | 67 | 0.667 |
| nda-12 | 74.3% | 101 | 0.907 |
| nda-13 | 72.7% | 121 | 0.815 |
| nda-17 | 69.6% | 23 | 0.788 |
| nda-11 | 66.7% | 9 | 0.857 |
| nda-18 | 56.7% | 30 | 0.737 |
| nda-16 | 43.1% | 51 | 0.638 |
| nda-15 | 35.3% | 68 | 0.806 |
| nda-19 | 30.2% | 86 | 0.676 |
| nda-20 | 30.2% | 63 | 0.455 |

## 0.5 Decision rule

- worst quartile by evidence F1 (4 hypotheses): ['nda-10', 'nda-16', 'nda-20', 'nda-7']
- (c) fabricated + (a) missed = **14.3%** of all failures  (threshold ≥40% → PROCEED)
- (d) partial + (e) wrong span **within the worst quartile** = **16.1%** of all failures

### BRANCH TAKEN: **SKIP Phases 1 and 2**

