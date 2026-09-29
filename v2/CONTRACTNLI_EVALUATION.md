# ContractNLI — detailed evaluation (v2)

ContractNLI only. CUAD, risk notes, ablations and windowing are deliberately excluded.

Rescored offline from archived v2 predictions; no inference was re-run.

The dataset's third class is **NotMentioned**, which is the Neutral class in NLI terms. It is reported under its dataset name below.


## Sanity checks (stated first, because everything below depends on them)

| check | result |
|---|---|
| evaluation split | **dev** (`v2/combined/dev.jsonl`) |
| dev or test | dev — the test split is untouched for these four models |
| ContractNLI examples per model | 1037 |
| identical example set across all four models | YES — gold labels AND gold evidence identical element-wise in file order |
| rows excluded | none; every row is scored |
| unparseable outputs | counted as a wrong prediction, never dropped |

| model | examples | unparseable |
|---|---|---|
| Qwen3-4B | 1037 | 0 |
| Qwen3-8B | 1037 | 0 |
| Llama-3.1-8B | 1037 | 0 |
| Qwen3-14B | 1037 | 0 |

Unparseable outputs are scored as errors rather than removed. Dropping them would flatter a model that fails loudly.


## 1. Overall verdict classification

Macro precision and macro recall are unweighted means over the three classes, so the rare Contradiction class counts as much as the common ones.

| model | accuracy | macro-F1 | macro precision | macro recall | 95% CI (macro-F1) |
|---|---|---|---|---|---|
| Qwen3-4B | 0.869 | **0.824** | 0.809 | 0.847 | [0.793, 0.852] |
| Qwen3-8B | 0.872 | **0.826** | 0.809 | 0.852 | [0.795, 0.853] |
| Llama-3.1-8B | 0.889 | **0.847** | 0.834 | 0.865 | [0.817, 0.874] |
| Qwen3-14B | 0.884 | **0.837** | 0.824 | 0.856 | [0.806, 0.865] |

## 2. Per-label results

Support: Entailment 519, Contradiction 95, NotMentioned 423 (total 1037).


### Entailment (support 519)

| model | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.902 | 0.917 | 0.909 | 476 | 52 | 43 |
| Qwen3-8B | 0.910 | 0.917 | 0.914 | 476 | 47 | 43 |
| Llama-3.1-8B | 0.915 | 0.938 | 0.927 | 487 | 45 | 32 |
| Qwen3-14B | 0.915 | 0.934 | 0.925 | 485 | 45 | 34 |

### Contradiction (support 95)

| model | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.623 | 0.800 | 0.700 | 76 | 46 | 19 |
| Qwen3-8B | 0.616 | 0.811 | 0.700 | 77 | 48 | 18 |
| Llama-3.1-8B | 0.670 | 0.811 | 0.733 | 77 | 38 | 18 |
| Qwen3-14B | 0.641 | 0.789 | 0.708 | 75 | 42 | 20 |

### NotMentioned (support 423)

| model | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.902 | 0.825 | 0.862 | 349 | 38 | 74 |
| Qwen3-8B | 0.902 | 0.830 | 0.865 | 351 | 38 | 72 |
| Llama-3.1-8B | 0.918 | 0.846 | 0.881 | 358 | 32 | 65 |
| Qwen3-14B | 0.915 | 0.844 | 0.878 | 357 | 33 | 66 |

## 4. Confusion matrices (rows = gold, columns = predicted)


**Qwen3-4B**

| gold \ pred | Entailment | Contradiction | NotMentioned | unparseable | total |
|---|---|---|---|---|---|
| **Entailment** | 476 | 15 | 28 | 0 | 519 |
| **Contradiction** | 9 | 76 | 10 | 0 | 95 |
| **NotMentioned** | 43 | 31 | 349 | 0 | 423 |

**Qwen3-8B**

| gold \ pred | Entailment | Contradiction | NotMentioned | unparseable | total |
|---|---|---|---|---|---|
| **Entailment** | 476 | 13 | 30 | 0 | 519 |
| **Contradiction** | 10 | 77 | 8 | 0 | 95 |
| **NotMentioned** | 37 | 35 | 351 | 0 | 423 |

**Llama-3.1-8B**

| gold \ pred | Entailment | Contradiction | NotMentioned | unparseable | total |
|---|---|---|---|---|---|
| **Entailment** | 487 | 10 | 22 | 0 | 519 |
| **Contradiction** | 8 | 77 | 10 | 0 | 95 |
| **NotMentioned** | 37 | 28 | 358 | 0 | 423 |

**Qwen3-14B**

| gold \ pred | Entailment | Contradiction | NotMentioned | unparseable | total |
|---|---|---|---|---|---|
| **Entailment** | 485 | 11 | 23 | 0 | 519 |
| **Contradiction** | 10 | 75 | 10 | 0 | 95 |
| **NotMentioned** | 35 | 31 | 357 | 0 | 423 |

## 3. Failure pattern per label


### Entailment

- **Qwen3-4B** P 0.902 / R 0.917. Missed 43 gold Entailment -> predicted as [NotMentioned 28, Contradiction 15]. 52 false Entailment came from [NotMentioned 43, Contradiction 9].
- **Qwen3-8B** P 0.910 / R 0.917. Missed 43 gold Entailment -> predicted as [NotMentioned 30, Contradiction 13]. 47 false Entailment came from [NotMentioned 37, Contradiction 10].
- **Llama-3.1-8B** P 0.915 / R 0.938. Missed 32 gold Entailment -> predicted as [NotMentioned 22, Contradiction 10]. 45 false Entailment came from [NotMentioned 37, Contradiction 8].
- **Qwen3-14B** P 0.915 / R 0.934. Missed 34 gold Entailment -> predicted as [NotMentioned 23, Contradiction 11]. 45 false Entailment came from [NotMentioned 35, Contradiction 10].

### Contradiction

- **Qwen3-4B** P 0.623 / R 0.800. Missed 19 gold Contradiction -> predicted as [NotMentioned 10, Entailment 9]. 46 false Contradiction came from [NotMentioned 31, Entailment 15].
- **Qwen3-8B** P 0.616 / R 0.811. Missed 18 gold Contradiction -> predicted as [Entailment 10, NotMentioned 8]. 48 false Contradiction came from [NotMentioned 35, Entailment 13].
- **Llama-3.1-8B** P 0.670 / R 0.811. Missed 18 gold Contradiction -> predicted as [NotMentioned 10, Entailment 8]. 38 false Contradiction came from [NotMentioned 28, Entailment 10].
- **Qwen3-14B** P 0.641 / R 0.789. Missed 20 gold Contradiction -> predicted as [NotMentioned 10, Entailment 10]. 42 false Contradiction came from [NotMentioned 31, Entailment 11].

### NotMentioned

- **Qwen3-4B** P 0.902 / R 0.825. Missed 74 gold NotMentioned -> predicted as [Entailment 43, Contradiction 31]. 38 false NotMentioned came from [Entailment 28, Contradiction 10].
- **Qwen3-8B** P 0.902 / R 0.830. Missed 72 gold NotMentioned -> predicted as [Entailment 37, Contradiction 35]. 38 false NotMentioned came from [Entailment 30, Contradiction 8].
- **Llama-3.1-8B** P 0.918 / R 0.846. Missed 65 gold NotMentioned -> predicted as [Entailment 37, Contradiction 28]. 32 false NotMentioned came from [Entailment 22, Contradiction 10].
- **Qwen3-14B** P 0.915 / R 0.844. Missed 66 gold NotMentioned -> predicted as [Entailment 35, Contradiction 31]. 33 false NotMentioned came from [Entailment 23, Contradiction 10].

## 5. Comparing the four models

Class-F1 side by side, so the source of any difference is visible rather than hidden in the macro average.

| class | Qwen3-4B | Qwen3-8B | Llama-3.1-8B | Qwen3-14B | spread |
|---|---|---|---|---|---|
| Entailment | 0.909 | 0.914 | 0.927 | 0.925 | 0.017 |
| Contradiction | 0.700 | 0.700 | 0.733 | 0.708 | 0.033 |
| NotMentioned | 0.862 | 0.865 | 0.881 | 0.878 | 0.019 |

### Paired bootstrap on macro-F1

Same dev rows for both models, 2000 resamples, percentile CI on the difference. A CI spanning zero means the gap is not separable from sampling noise. **This tests row sampling only; it does NOT cover training-seed variance.**

| A vs B | mean diff | 95% CI | P(A better) | separated |
|---|---|---|---|---|
| Qwen3-4B vs Qwen3-8B | -0.0023 | [-0.0204, +0.0152] | 0.419 | no |
| Qwen3-4B vs Llama-3.1-8B | -0.0233 | [-0.0405, -0.0070] | 0.002 | YES |
| Qwen3-4B vs Qwen3-14B | -0.0131 | [-0.0319, +0.0055] | 0.080 | no |
| Qwen3-8B vs Llama-3.1-8B | -0.0210 | [-0.0404, -0.0027] | 0.014 | YES |
| Qwen3-8B vs Qwen3-14B | -0.0108 | [-0.0277, +0.0057] | 0.108 | no |
| Llama-3.1-8B vs Qwen3-14B | +0.0102 | [-0.0086, +0.0292] | 0.851 | no |

## 6. Evidence extraction

Gold evidence-bearing examples: **614** of 1037. Gold evidence spans: **1227**.

| model | strict F1 | token F1 | token P | token R | span recall @J0.5 | spans located |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.731 | 0.801 | 0.852 | 0.808 | 0.723 | 887 |
| Qwen3-8B | 0.739 | 0.804 | 0.848 | 0.818 | 0.727 | 892 |
| Llama-3.1-8B | 0.747 | 0.812 | 0.856 | 0.823 | 0.736 | 903 |
| Qwen3-14B | 0.763 | 0.825 | 0.864 | 0.838 | 0.750 | 920 |

### Evidence failure decomposition

| model | strict failures | boundary only | partial | not found |
|---|---|---|---|---|
| Qwen3-4B | 218 | 18 | 140 | 60 |
| Qwen3-8B | 212 | 21 | 135 | 56 |
| Llama-3.1-8B | 209 | 17 | 135 | 57 |
| Qwen3-14B | 195 | 21 | 121 | 53 |

## 7. Long-context effect

ContractNLI prompts measured with the evaluation tokenizer: p50 2,272, p99 6,506, max 6,520 tokens against an input budget of 15,360.

**0 of 1037 ContractNLI dev prompts exceed the budget.** The context window never binds on this task, so there is no truncation effect to report and no evidence is rendered invisible.

Accuracy by prompt length, included because 'the window never binds' is not the same as 'length does not matter':

| bucket | examples | Qwen3-4B | Qwen3-8B | Llama-3.1-8B | Qwen3-14B |
|---|---|---|---|---|---|
| <2K | 442 | 0.860 | 0.851 | 0.882 | 0.867 |
| 2-4K | 459 | 0.876 | 0.887 | 0.898 | 0.898 |
| 4-8K | 136 | 0.875 | 0.890 | 0.882 | 0.897 |