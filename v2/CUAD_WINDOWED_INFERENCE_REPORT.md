# CUAD windowed inference — A (direct 16K) vs B (sliding window)

## 1. Checkpoint used

- **Base**: `Qwen/Qwen3-4B`
- **Adapter**: `/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final`
- **Adapter sha256**: `f9c989992dc19f36bbce5e79dda3eacb0544488453d6e45f09a597ee83633bc6`

This is the multitask 4B checkpoint, trained on the three-task dataset (ContractNLI + CUAD + risk notes) at max_seq_len 16,384.

## 2. Confirmation that A and B use the same checkpoint

Verified three ways, not assumed:

1. A's metrics file records `base` = `Qwen/Qwen3-4B` and `adapter` = `/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final`.
2. A's SLURM log line reads `=== eval Qwen/Qwen3-4B adapter=/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final ===`, so the path in the metrics file is the path the job actually loaded.
3. B was launched against the identical path, recorded in its own meta file as `/scratch/ibi761/legalai/v2_sft/Qwen__Qwen3-4B__3task/final`.

**Match: YES.** A and B are the same weights.

## 3. Direct 16K baseline source

- Predictions: `/scratch/ibi761/legalai/v2_results/raw__q4_3task.jsonl`
- Metrics: `/scratch/ibi761/legalai/v2_results/metrics__q4_3task.json`
- A was **not** re-run. n = 2489 rows total (1098 CUAD), wall clock 100.6 min, 198 prompts left-truncated.
- Join key is `(contract, category)`, unique in both files, and the two key sets are identical, so every comparison below is like-for-like on the same 1098 pairs.

## 4. Windowing configuration

| setting | value |
|---|---|
| total model context | 16,384 |
| reserved for generation | 1,024 |
| input budget | 15,360 |
| measured scaffold overhead | 108 |
| contract fits in one call if | <= 15,252 tokens |
| window content tokens | 14,000 |
| stride | 12,000 |
| overlap | 2,000 |

**CASE 1 is honoured.** A contract is windowed only when the complete prompt exceeds the input budget. Short contracts are passed whole in a single call and are byte-identical to what A saw, so they cannot manufacture a difference. The overhead figure is measured by rendering the template with an empty contract and the longest category name, not guessed.

**Aggregation.** A window's `present: true` counts only if its emitted span is grounded in that window via `reasoning_filters._grounded`. Contract-level present if ANY window casts a grounded vote. No thresholds, no calibration, no per-category tuning.

## 5. Cost pre-check

| quantity | value |
|---|---|
| contracts requiring multiple windows | 11 |
| contracts evaluated in one window | 50 |
| total model calls, B | 1,440 |
| total model calls, A | 1,098 |
| average windows per contract | 1.31 |
| maximum windows for any contract | 5 |
| windows-per-contract histogram | {'1': 50, '2': 6, '3': 3, '4': 1, '5': 1} |

Pre-check estimate was under two hours, so B was launched without asking. Actual: **43.4 min**, 1.809s per call.

## 6-7. Direct 16K results and windowed results

| metric | A direct 16K | B windowed | delta |
|---|---|---|---|
| PRESENT F1 | 0.7988 | 0.8222 | +0.0233 |
| PRESENT precision | 0.8787 | 0.8812 | +0.0026 |
| PRESENT recall | 0.7322 | 0.7705 | +0.0383 |
| evidence recall | 0.3907 | 0.3880 | -0.0027 |
| TP | 268 | 282 | +14 |
| FP | 37 | 38 | +1 |
| FN | 98 | 84 | -14 |
| TN | 695 | 694 | -1 |

n = 1098 (contract, category) pairs; evidence recall computed over the 366 pairs that carry gold evidence.

## 8. Length-stratified comparison

Windowing can only act where the window binds. Most dev contracts fit in one call, so the pooled numbers above are diluted and this table is the one that carries the answer.

| bucket | contracts | pairs | A F1 | B F1 | A recall | B recall | A ev-recall | B ev-recall |
|---|---|---|---|---|---|---|---|---|
| <16K | 51 | 918 | 0.818 | 0.817 | 0.770 | 0.758 | 0.449 | 0.442 |
| 16-32K | 8 | 144 | 0.734 | 0.833 | 0.622 | 0.793 | 0.268 | 0.232 |
| >32K | 2 | 36 | 0.788 | 0.842 | 0.684 | 0.842 | 0.105 | 0.316 |

## 9. Per-category comparison (complete)

| category | pairs | gold+ | A F1 | B F1 | A rec | B rec | A ev-rec | B ev-rec |
|---|---|---|---|---|---|---|---|---|
| Anti-Assignment | 61 | 44 | 0.966 | 0.977 | 0.955 | 0.977 | 0.591 | 0.591 |
| Audit Rights | 61 | 28 | 0.880 | 0.868 | 0.786 | 0.821 | 0.393 | 0.357 |
| Cap On Liability | 61 | 29 | 0.881 | 0.881 | 0.897 | 0.897 | 0.414 | 0.379 |
| Change Of Control | 61 | 16 | 0.733 | 0.643 | 0.688 | 0.562 | 0.312 | 0.250 |
| Covenant Not To Sue | 61 | 10 | 0.625 | 0.625 | 0.500 | 0.500 | 0.200 | 0.300 |
| Exclusivity | 61 | 20 | 0.765 | 0.833 | 0.650 | 0.750 | 0.300 | 0.350 |
| Insurance | 61 | 20 | 0.824 | 0.919 | 0.700 | 0.850 | 0.400 | 0.450 |
| Ip Ownership Assignment | 61 | 13 | 0.700 | 0.818 | 0.538 | 0.692 | 0.231 | 0.231 |
| License Grant | 61 | 29 | 0.862 | 0.877 | 0.862 | 0.862 | 0.379 | 0.345 |
| Minimum Commitment | 61 | 20 | 0.516 | 0.563 | 0.400 | 0.450 | 0.100 | 0.150 |
| Non-Compete | 61 | 15 | 0.667 | 0.720 | 0.533 | 0.600 | 0.067 | 0.067 |
| Non-Transferable License | 61 | 12 | 0.720 | 0.769 | 0.750 | 0.833 | 0.583 | 0.583 |
| Notice Period To Terminate Renewal | 61 | 14 | 0.833 | 0.783 | 0.714 | 0.643 | 0.571 | 0.429 |
| Post-Termination Services | 61 | 21 | 0.722 | 0.757 | 0.619 | 0.667 | 0.238 | 0.190 |
| Renewal Term | 61 | 24 | 0.818 | 0.870 | 0.750 | 0.833 | 0.667 | 0.708 |
| Revenue/Profit Sharing | 61 | 20 | 0.833 | 0.919 | 0.750 | 0.850 | 0.250 | 0.250 |
| Termination For Convenience | 61 | 21 | 0.791 | 0.791 | 0.810 | 0.810 | 0.571 | 0.619 |
| Uncapped Liability | 61 | 10 | 0.500 | 0.476 | 0.500 | 0.500 | 0.300 | 0.300 |

### Truncation-sensitive categories

| category | A F1 | B F1 | A ev-recall | B ev-recall | verdict |
|---|---|---|---|---|---|
| Non-Transferable License | 0.720 | 0.769 | 0.583 | 0.583 | no change |
| Ip Ownership Assignment | 0.700 | 0.818 | 0.231 | 0.231 | no change |
| Exclusivity | 0.765 | 0.833 | 0.300 | 0.350 | recovered |
| Non-Compete | 0.667 | 0.720 | 0.067 | 0.067 | no change |

## 10. Evidence visibility and coverage

| | A direct 16K | B windowed |
|---|---|---|
| gold evidence spans | 745 | 745 |
| spans visible to the model | 653 | 745 |
| spans invisible | 92 | 0 |
| visibility | 87.65% | 100.00% |
| empirical evidence-recall ceiling | 87.65% | 100.00% |

B's visibility is **measured**, not assumed: every gold span was searched for in the windows actually constructed, using the same `_grounded` test the aggregation uses.

## 11. Coordinate-mapping validation

- Gold spans round-tripped: **745 / 745**, failures 0, unlocatable 0.
- Method: map the span to original-contract offsets, then slice the ORIGINAL contract at those offsets and require the result to match.
- Positions exercised by gold spans: 31 near a window start, 78 near a window end, 0 inside an overlap region.

**No gold span landed in an overlap region**, so that branch was driven directly with real contract text sampled from the shared regions themselves. A span in an overlap is visible from two windows at different local offsets and both must resolve to the same original coordinates:

| check | count |
|---|---|
| overlap regions exercised | 19 |
| spans tested | 227 |
| both windows agree on offsets | 227 |
| both windows disagree | 0 |
| round-trip OK | 227 |
| round-trip FAIL | 0 |

During B itself, 0 spans passed the groundedness test but could not be assigned coordinates.

## 12. Inference cost

| | A | B |
|---|---|---|
| model calls (CUAD) | 1,098 | 1,440 |
| ratio | 1.00x | 1.31x |
| wall clock | 100.6 min (all 3 tasks, 2489 rows) | 43.4 min (CUAD only) |
| sec per call | 2.425 | 1.809 |
| avg windows/contract | 1.00 | 1.31 |
| max windows/contract | 1 | 5 |

Prefix caching was enabled for both. In B the system prompt plus the contract window is a shared prefix across all 18 categories of a contract, and prompts were ordered by (contract, window) so that prefix stays hot. A's wall clock covers all three tasks and is not a per-call comparison; the per-call figures are.

## 13. Error analysis

| class | definition | count |
|---|---|---|
| A. coverage | gold evidence outside the visible 16K input (A misses) | 21 |
| B. model accuracy | evidence was visible, model still wrong (A misses) | 77 |
| C. aggregation | B says present on a gold-absent pair via a grounded window vote | 38 |
| D. grounding | window claimed present but cited text not in that window | 20 |

Class D is the one the aggregation rule exists to stop. Each of those 20 claims would have become a contract-level false positive under a plain OR over window verdicts.

- **A (coverage)** examples: `Insurance` in `ALLIANCEBANCORPINCOFPENNSYLVANIA_10_18_2`; `Covenant Not To Sue` in `DovaPharmaceuticalsInc_20181108_10-Q_EX-`; `Non-Compete` in `DovaPharmaceuticalsInc_20181108_10-Q_EX-`
- **B (model accuracy)** examples: `Non-Compete` in `2ThemartComInc_19990826_10-12G_EX-10.10_`; `Uncapped Liability` in `2ThemartComInc_19990826_10-12G_EX-10.10_`; `Post-Termination Services` in `ALLIANCEBANCORPINCOFPENNSYLVANIA_10_18_2`
- **C (aggregation)** examples: `Termination For Convenience` in `2ThemartComInc_19990826_10-12G_EX-10.10_` (1/1 windows voted); `Cap On Liability` in `ALLIANCEBANCORPINCOFPENNSYLVANIA_10_18_2` (2/3 windows voted); `Termination For Convenience` in `ALLIANCEBANCORPINCOFPENNSYLVANIA_10_18_2` (1/3 windows voted)
- **D (grounding)** examples: `Change Of Control` in `Columbia Laboratories, (Bermuda) Ltd. - ` (1 ungrounded); `Post-Termination Services` in `Columbia Laboratories, (Bermuda) Ltd. - ` (1 ungrounded); `Cap On Liability` in `CybergyHoldingsInc_20140520_10-Q_EX-10.2` (1 ungrounded)
