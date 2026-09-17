# Training diagnostics

Loss curves and the signals behind rank / batch / LR decisions.

**How to read**: loss flat at the end = converged; train≈eval and both flat
= capacity-bound, raise rank; grad-norm spikes = LR too high; high late-run

## All runs

![all](training_curves/eval_all.png)

loss variance = effective batch too small.


## Qwen__Qwen3-14B

![Qwen__Qwen3-14B](training_curves/loss_Qwen__Qwen3-14B.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0329 / eval 0.0357 |
| cost | 94 min, peak 49.18 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — eval rose 12% off its minimum — OVERFITTING, fewer epochs or lower rank
- **learning rate** — grad-norm stable (median 0.144, max 0.473)
- **batch size** — late-run loss CV 0.11 — stable


## Qwen__Qwen3-14B__amatch

![Qwen__Qwen3-14B__amatch](training_curves/loss_Qwen__Qwen3-14B__amatch.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0429 / eval 0.0631 |
| cost | 35 min, peak 49.14 GB |

**Diagnosis**

- **convergence** — STILL FALLING — undertrained, more epochs would help
- **rank/epochs** — eval rose 27% off its minimum — OVERFITTING, fewer epochs or lower rank
- **learning rate** — grad-norm stable (median 0.130, max 0.320)
- **batch size** — late-run loss CV 0.03 — stable


## Qwen__Qwen3-14B__armb

![Qwen__Qwen3-14B__armb](training_curves/loss_Qwen__Qwen3-14B__armb.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.1122 / eval 0.4497 |
| cost | 38 min, peak 49.16 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — eval rose 33% off its minimum — OVERFITTING, fewer epochs or lower rank
- **learning rate** — grad-norm stable (median 0.151, max 0.240)
- **batch size** — late-run loss CV 0.06 — stable


## Qwen__Qwen3-4B

![Qwen__Qwen3-4B](training_curves/loss_Qwen__Qwen3-4B.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0395 / eval 0.0365 |
| cost | 49 min, peak 25.23 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — healthy train/eval gap
- **learning rate** — grad-norm stable (median 0.233, max 1.269)
- **batch size** — late-run loss CV 0.11 — stable


## Qwen__Qwen3-4B__amatch

![Qwen__Qwen3-4B__amatch](training_curves/loss_Qwen__Qwen3-4B__amatch.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0525 / eval 0.0519 |
| cost | 18 min, peak 25.23 GB |

**Diagnosis**

- **convergence** — STILL FALLING — undertrained, more epochs would help
- **rank/epochs** — healthy train/eval gap
- **learning rate** — grad-norm stable (median 0.207, max 0.679)
- **batch size** — late-run loss CV 0.04 — stable


## Qwen__Qwen3-4B__armb

![Qwen__Qwen3-4B__armb](training_curves/loss_Qwen__Qwen3-4B__armb.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.1375 / eval 0.3630 |
| cost | 19 min, peak 25.23 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — eval rose 11% off its minimum — OVERFITTING, fewer epochs or lower rank
- **learning rate** — grad-norm stable (median 0.230, max 0.320)
- **batch size** — late-run loss CV 0.04 — stable


## Qwen__Qwen3-8B

![Qwen__Qwen3-8B](training_curves/loss_Qwen__Qwen3-8B.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0396 / eval 0.0361 |
| cost | 64 min, peak 34.64 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — healthy train/eval gap
- **learning rate** — grad-norm stable (median 0.167, max 0.782)
- **batch size** — late-run loss CV 0.10 — stable


## Qwen__Qwen3-8B__amatch

![Qwen__Qwen3-8B__amatch](training_curves/loss_Qwen__Qwen3-8B__amatch.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0524 / eval 0.0560 |
| cost | 24 min, peak 34.64 GB |

**Diagnosis**

- **convergence** — STILL FALLING — undertrained, more epochs would help
- **rank/epochs** — healthy train/eval gap
- **learning rate** — grad-norm stable (median 0.188, max 0.585)
- **batch size** — late-run loss CV 0.07 — stable


## Qwen__Qwen3-8B__armb

![Qwen__Qwen3-8B__armb](training_curves/loss_Qwen__Qwen3-8B__armb.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.1255 / eval 0.4067 |
| cost | 25 min, peak 34.64 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — eval rose 13% off its minimum — OVERFITTING, fewer epochs or lower rank
- **learning rate** — grad-norm stable (median 0.167, max 0.252)
- **batch size** — late-run loss CV 0.05 — stable


## google__gemma-4-12B-it

![google__gemma-4-12B-it](training_curves/loss_google__gemma-4-12B-it.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0497 / eval 0.0292 |
| cost | 177 min, peak 58.57 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — healthy train/eval gap
- **learning rate** — 3 grad-norm spikes >5x median — LR TOO HIGH
- **batch size** — late-run loss CV 0.11 — stable


## google__gemma-4-12B-it__cw3

![google__gemma-4-12B-it__cw3](training_curves/loss_google__gemma-4-12B-it__cw3.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0490 / eval 0.0270 |
| cost | 176 min, peak 58.57 GB |

**Diagnosis**

- **convergence** — FLAT at the end — converged or capacity-bound
- **rank/epochs** — healthy train/eval gap
- **learning rate** — 5 grad-norm spikes >5x median — LR TOO HIGH
- **batch size** — late-run loss CV 0.13 — stable


## google__gemma-4-12B-it__os4

![google__gemma-4-12B-it__os4](training_curves/loss_google__gemma-4-12B-it__os4.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0467 / eval 0.0333 |
| cost | 240 min, peak 58.57 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — healthy train/eval gap
- **learning rate** — 13 grad-norm spikes >5x median — LR TOO HIGH
- **batch size** — late-run loss CV 0.23 — stable


## ibm-granite__granite-4.2-8b

![ibm-granite__granite-4.2-8b](training_curves/loss_ibm-granite__granite-4.2-8b.png)

| | |
|---|---|
| LoRA | r=16 alpha=32 dropout=0.05 |
| optim | lr=0.0001 cosine warmup=0.03 epochs=1.0 |
| batch | 1 x 16 = 16 |
| final | train 0.0311 / eval 0.0364 |
| cost | 67 min, peak 31.2 GB |

**Diagnosis**

- **convergence** — RISING at the end — overfitting or LR too high
- **rank/epochs** — healthy train/eval gap
- **learning rate** — grad-norm stable (median 0.352, max 1.172)
- **batch size** — late-run loss CV 0.12 — stable

