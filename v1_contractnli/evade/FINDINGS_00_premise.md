# EVADE task — premise correction, recorded before any run

## The 62% cut was mostly NOT the agreement filter

The task motivation attributed the 7,188 → 2,733 reduction to blind verdict
agreement. Measured:

| stage | rows | cut | share of total loss |
|---|---|---|---|
| start | 7,188 | — | — |
| after blind **agreement** filter | 5,736 | −1,452 (20.2%) | **32.6%** |
| after the four **string** filters | 2,733 | −3,003 | **67.4%** |

**Two thirds of the loss came from the string filters, not from agreement.**
The dominant rejections were `f1_ungrounded_quote` (~21% of all rows) and
`f2_evidence_disjoint` (~21%) — both rationale-QUALITY checks.

## Why this matters, and it is a result rather than a bookkeeping note

Those two filters test whether the teacher **wrote a clean rationale**: did it
quote text that appears in the contract, and did its cited evidence overlap
gold. Neither has anything to do with whether the row's LABEL is correct. A row
can be perfectly labelled and rejected because the teacher paraphrased inside
quotation marks; a row can be mislabelled and kept because the teacher happened
to quote fluently.

So the old pipeline's majority filter was **selecting for teacher articulacy,
not for data correctness** — which is close to the opposite of what a
data-cleaning filter should do.

## It re-frames the Arm B comparison

Arm B trained on those 2,733 rows. That set is therefore **biased toward rows
the teacher could write cleanly about**, not a random or a quality-selected
sample. Arm B's loss against the matched control (−0.054 to −0.059 macro-F1)
was measured on rows drawn from that biased pool for BOTH arms, so the
comparison itself remains valid — the subset-matched control used the identical
2,733 rows. But any claim about Arm B generalising to the full corpus inherits
the bias, and the 62% figure should not be described as label filtering.

## Consequence for this task

EVADE's (a) "does the rationale reference real contract text" and (b) "does its
logic support the label" are close cousins of the old `f1`/`f2` string filters —
both are rationale-quality checks. The genuinely new signal is **(c): is the
assigned label defensible at all**. That is the only question of the three that
is about the DATA rather than about the teacher's writing, and it is the one to
watch in the results.

Arm 4 is therefore split:
- **arm 4a — 5,736 rows**, agreement filter only. Isolates agreement filtering,
  the conceptual comparison.
- **arm 4b — 2,733 rows**, agreement + string filters. The historical set,
  already trained.
