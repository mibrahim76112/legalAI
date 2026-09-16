#!/usr/bin/env python3
"""
Training diagnostics for every SFT run: loss curves + the signals that tell you
whether to change rank, batch size, or learning rate.

Reads manifest.json (log_history already carries loss, eval_loss, grad_norm,
learning_rate, epoch) and emits one PNG per run plus a combined comparison, and
a markdown verdict per run.

How to read each panel:
  loss          - is it still falling at the end (undertrained) or flat
                  (converged / capacity-bound)?
  train vs eval - eval above train and rising = overfitting; both flat and
                  equal = underfitting, which is the RANK signal
  grad_norm     - spikes or a rising trend = LR too high; near-zero early =
                  LR too low or already converged
  LR schedule   - confirms warmup and cosine decay actually ran
  loss variance - high step-to-step scatter at fixed LR = BATCH too small
"""
import json, glob, argparse, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(h):
    tr = [(x["step"], x["loss"]) for x in h if "loss" in x]
    ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
    gn = [(x["step"], x["grad_norm"]) for x in h if "grad_norm" in x and x["grad_norm"] is not None]
    lr = [(x["step"], x["learning_rate"]) for x in h if "learning_rate" in x]
    return tr, ev, gn, lr


def diagnose(tr, ev, gn):
    """Turn the curves into the three decisions the user actually makes."""
    out = []
    ys = [y for _, y in tr]
    tail = ys[max(1, int(len(ys) * 0.75)):]
    head = ys[:max(1, int(len(ys) * 0.25))]
    # convergence
    slope = (st.mean(tail) - st.mean(head)) / max(abs(st.mean(head)), 1e-9)
    last_q = ys[max(1, int(len(ys) * 0.9)):]
    drift = (last_q[-1] - last_q[0]) / max(abs(last_q[0]), 1e-9) if len(last_q) > 1 else 0
    if abs(drift) < 0.05:
        out.append(("convergence", "FLAT at the end — converged or capacity-bound"))
    elif drift < 0:
        out.append(("convergence", "STILL FALLING — undertrained, more epochs would help"))
    else:
        out.append(("convergence", "RISING at the end — overfitting or LR too high"))
    # overfit / rank
    if ev:
        e = [y for _, y in ev]
        if len(e) > 2 and e[-1] > min(e) * 1.10:
            out.append(("rank/epochs", f"eval rose {100*(e[-1]/min(e)-1):.0f}% off its "
                                       f"minimum — OVERFITTING, fewer epochs or lower rank"))
        elif st.mean(tail) > 0 and abs(st.mean(tail) - e[-1]) / max(e[-1], 1e-9) < 0.15:
            out.append(("rank/epochs", "train ~= eval and both flat — UNDERFITTING is "
                                       "possible; try higher rank before more epochs"))
        else:
            out.append(("rank/epochs", "healthy train/eval gap"))
    # LR
    if gn:
        g = [y for _, y in gn]
        gmax, gmed = max(g), st.median(g)
        spikes = sum(1 for y in g if y > 5 * gmed)
        if spikes > max(1, len(g) * 0.05):
            out.append(("learning rate", f"{spikes} grad-norm spikes >5x median — LR TOO HIGH"))
        elif gmed < 0.05:
            out.append(("learning rate", f"grad-norm median {gmed:.3f} very low — LR could go up"))
        else:
            out.append(("learning rate", f"grad-norm stable (median {gmed:.3f}, max {gmax:.3f})"))
    # batch
    if len(tail) > 3:
        cv = st.pstdev(tail) / max(abs(st.mean(tail)), 1e-9)
        if cv > 0.35:
            out.append(("batch size", f"late-run loss CV {cv:.2f} — noisy, a LARGER "
                                      f"effective batch would stabilise it"))
        else:
            out.append(("batch size", f"late-run loss CV {cv:.2f} — stable"))
    return out


def plot_run(name, m, outdir):
    tr, ev, gn, lr = series(m["log_history"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"{name}   r={m['lora']['r']} alpha={m['lora']['alpha']} "
                 f"lr={m['optim']['lr']} eff_batch={m['optim']['effective_batch']} "
                 f"seed={m['seed']}", fontsize=10)
    a = ax[0][0]
    a.plot(*zip(*tr), lw=1.2, label="train")
    if ev: a.plot(*zip(*ev), "o-", ms=4, lw=1.2, label="eval")
    a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("loss (log)")
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_title("loss", fontsize=9)
    a = ax[0][1]
    ys = [y for _, y in tr]; lo = max(1, int(len(ys) * .2))
    a.plot([s for s, _ in tr][lo:], ys[lo:], lw=1.2)
    if ev: a.plot(*zip(*[e for e in ev if e[0] >= tr[lo][0]]), "o-", ms=4)
    a.set_xlabel("step"); a.set_ylabel("loss"); a.grid(alpha=.3)
    a.set_title("loss, tail (linear) — is it still moving?", fontsize=9)
    a = ax[1][0]
    if gn: a.plot(*zip(*gn), lw=1.2, color="tab:red")
    a.set_xlabel("step"); a.set_ylabel("grad norm"); a.grid(alpha=.3)
    a.set_title("grad norm — spikes mean LR too high", fontsize=9)
    a = ax[1][1]
    if lr: a.plot(*zip(*lr), lw=1.2, color="tab:green")
    a.set_xlabel("step"); a.set_ylabel("lr"); a.grid(alpha=.3)
    a.set_title("LR schedule (warmup + cosine)", fontsize=9)
    fig.tight_layout()
    p = Path(outdir) / f"loss_{name}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, diagnose(tr, ev, gn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", default="/scratch/ibi761/legalai/sft_out")
    ap.add_argument("--out", default="training_curves")
    ap.add_argument("--report", default="TRAINING_DIAGNOSTICS.md")
    args = ap.parse_args()
    Path(args.out).mkdir(exist_ok=True)

    L = ["# Training diagnostics\n",
         "Loss curves and the signals behind rank / batch / LR decisions.\n",
         "**How to read**: loss flat at the end = converged; train≈eval and both flat",
         "= capacity-bound, raise rank; grad-norm spikes = LR too high; high late-run",
         "loss variance = effective batch too small.\n"]
    runs = []
    for mp in sorted(glob.glob(f"{args.sft_dir}/*/manifest.json")):
        name = Path(mp).parent.name
        m = json.load(open(mp))
        png, diag = plot_run(name, m, args.out)
        runs.append((name, m, png, diag))
        L.append(f"\n## {name}\n")
        L.append(f"![{name}]({png})\n")
        o = m["optim"]
        L.append(f"| | |\n|---|---|")
        L.append(f"| LoRA | r={m['lora']['r']} alpha={m['lora']['alpha']} "
                 f"dropout={m['lora']['dropout']} |")
        L.append(f"| optim | lr={o['lr']} {o['scheduler']} warmup={o['warmup_ratio']} "
                 f"epochs={o['epochs']} |")
        L.append(f"| batch | {o['batch_size']} x {o['grad_accum']} = {o['effective_batch']} |")
        L.append(f"| final | train {m['final_train_loss']:.4f} / eval "
                 f"{m['final_eval_loss']:.4f} |")
        L.append(f"| cost | {m['wall_seconds']/60:.0f} min, peak {m['peak_gpu_gb']} GB |\n")
        L.append("**Diagnosis**\n")
        for k, v in diag:
            L.append(f"- **{k}** — {v}")
        L.append("")

    # combined eval-loss comparison
    fig, a = plt.subplots(figsize=(8, 5))
    for name, m, _, _ in runs:
        ev = [(x["step"], x["eval_loss"]) for x in m["log_history"] if "eval_loss" in x]
        if ev: a.plot(*zip(*ev), "o-", ms=3, lw=1.2, label=name)
    a.set_xlabel("step"); a.set_ylabel("eval loss"); a.set_yscale("log")
    a.grid(alpha=.3); a.legend(fontsize=7); a.set_title("eval loss, all runs")
    p = Path(args.out) / "eval_all.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    L.insert(4, f"\n## All runs\n\n![all]({p})\n")

    Path(args.report).write_text("\n".join(L) + "\n")
    print(f"-> {args.report}  and {len(runs)+1} PNGs in {args.out}/")
    for name, _, _, diag in runs:
        print(f"\n{name}")
        for k, v in diag:
            print(f"   {k:16s} {v}")


if __name__ == "__main__":
    main()
