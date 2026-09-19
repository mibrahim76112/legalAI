# Phase 0 diagnostic — dev_Qwen3-14B split

Predictions: `raw____localscratch__ibi761.21892913.0__merged_Qwen_Qwen3-14B__nothink__dev__sft.jsonl`  (1037 rows)

Analysis only. **Dev split** — test is reserved and untouched.


## 0.1 Per-hypothesis breakdown

| hyp | n | gold E/C/NM | macro-F1 | acc | ev F1 | ev exact | ev partial |
|---|---|---|---|---|---|---|---|
| nda-20 | 61 | 13/12/36 | 0.572 | 0.541 | **0.448** | 0.393 | 0.810 |
| nda-18 | 61 | 11/0/50 | 0.593 | 0.934 | **0.632** | 0.885 | 0.720 |
| nda-7 | 61 | 39/14/8 | 0.686 | 0.803 | **0.711** | 0.574 | 0.751 |
| nda-16 | 61 | 22/2/37 | 0.585 | 0.869 | **0.727** | 0.803 | 0.761 |
| nda-3 | 61 | 45/0/16 | 0.585 | 0.902 | **0.730** | 0.672 | 0.706 |
| nda-10 | 61 | 29/0/32 | 0.590 | 0.885 | **0.731** | 0.770 | 0.736 |
| nda-19 | 61 | 43/0/18 | 0.570 | 0.885 | **0.737** | 0.672 | 0.767 |
| nda-1 | 61 | 19/10/32 | 0.748 | 0.820 | **0.750** | 0.803 | 0.694 |
| nda-17 | 61 | 9/8/44 | 0.793 | 0.869 | **0.765** | 0.869 | 0.765 |
| nda-4 | 61 | 50/3/8 | 0.719 | 0.902 | **0.783** | 0.672 | 0.783 |
| nda-2 | 61 | 6/44/11 | 0.554 | 0.836 | **0.791** | 0.705 | 0.786 |
| nda-5 | 61 | 54/2/5 | 0.546 | 0.934 | **0.816** | 0.705 | 0.806 |
| nda-15 | 61 | 39/0/22 | 0.594 | 0.902 | **0.838** | 0.803 | 0.829 |
| nda-13 | 61 | 47/0/14 | 0.667 | 1.000 | **0.840** | 0.787 | 0.872 |
| nda-11 | 61 | 8/0/53 | 0.613 | 0.967 | **0.857** | 0.967 | 0.750 |
| nda-12 | 61 | 40/0/21 | 0.654 | 0.984 | **0.892** | 0.869 | 0.930 |
| nda-8 | 61 | 45/0/16 | 0.652 | 0.984 | **0.930** | 0.902 | 0.908 |

## 0.3 Evidence-failure attribution

253 failures of 1037 rows (24.4%).

| category | n | % of failures | % of all rows |
|---|---|---|---|
| (a) missed — pred empty, gold non-empty | 33 | 13.0% | 3.2% |
| (b) over-extraction — pred non-empty, gold empty | 67 | 26.5% | 6.5% |
| (c) fabricated — span absent from contract | 4 | 1.6% | 0.4% |
| (d) partial — grounded, partial overlap with gold | 116 | 45.8% | 11.2% |
| (e) wrong span — grounded, no overlap | 33 | 13.0% | 3.2% |

## 0.3b How near are the near-misses? (item 8)

Gold spans recovered / required, over 186 gold-bearing failures:

| p10 | p25 | p50 | p75 | p90 | share ≥0.5 | share =0 |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.40 | 0.50 | 0.67 | 47.8% | 36.0% |

Strict evidence F1 (**primary, unchanged**): **0.7719**

Partial-credit variant (reported alongside, NOT a substitute): 0.7960


## 0.4 Positional check

Gold-evidence first-half share, per hypothesis (min 30.2%, max 96.0%):

| hyp | first-half share | n spans | ev F1 |
|---|---|---|---|
| nda-2 | 96.0% | 101 | 0.791 |
| nda-1 | 94.8% | 58 | 0.750 |
| nda-7 | 94.2% | 104 | 0.711 |
| nda-5 | 93.1% | 102 | 0.816 |
| nda-3 | 90.7% | 75 | 0.730 |
| nda-4 | 90.0% | 90 | 0.783 |
| nda-8 | 84.6% | 78 | 0.930 |
| nda-10 | 83.6% | 67 | 0.731 |
| nda-12 | 74.3% | 101 | 0.892 |
| nda-13 | 72.7% | 121 | 0.840 |
| nda-17 | 69.6% | 23 | 0.765 |
| nda-11 | 66.7% | 9 | 0.857 |
| nda-18 | 56.7% | 30 | 0.632 |
| nda-16 | 43.1% | 51 | 0.727 |
| nda-15 | 35.3% | 68 | 0.838 |
| nda-19 | 30.2% | 86 | 0.737 |
| nda-20 | 30.2% | 63 | 0.448 |

## 0.5 Decision rule

- worst quartile by evidence F1 (4 hypotheses): ['nda-16', 'nda-18', 'nda-20', 'nda-7']
- (c) fabricated + (a) missed = **14.6%** of all failures  (threshold ≥40% → PROCEED)
- (d) partial + (e) wrong span **within the worst quartile** = **14.2%** of all failures

### BRANCH TAKEN: **SKIP Phases 1 and 2**

