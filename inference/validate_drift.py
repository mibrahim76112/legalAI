"""Measure 4-bit drift: local MLX output vs the cluster's bf16 output.

Runs validation_60.jsonl through the local base + converted adapter and
reports agreement with `cluster_bf16_output` (isolates quantization) and,
for context only, accuracy vs gold on both sides. 60 rows is a smoke check,
not an evaluation.

    python inference/validate_drift.py --out results/drift_4bit.jsonl
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "macbundle" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doc_harness import parse_output, _find_json, strip_wrappers  # noqa: E402
from text_norm import norm  # noqa: E402
from engine import Engine  # noqa: E402
from remote import RemoteEngine  # noqa: E402


def parse_cuad(raw):
    obj = _find_json(strip_wrappers(raw))
    if not isinstance(obj, dict) or not isinstance(obj.get("present"), bool):
        return None, []
    ev = obj.get("evidence") or []
    if isinstance(ev, str):
        ev = [ev]
    return obj["present"], [e for e in ev if isinstance(e, str) and e.strip()]


def label_and_evidence(task, raw):
    if task == "contractnli":
        p = parse_output(raw)
        return p["verdict"], p["evidence"]
    if task == "cuad":
        return parse_cuad(raw)
    return None, []


def gold_label(row):
    return row["verdict"] if row["task"] == "contractnli" else row["present"]


def ev_key(ev):
    return tuple(norm(e) for e in ev)


def macro_f1(pairs, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def report(recs):
    print("\n=== agreement with cluster bf16 (quantization drift) ===")
    for task in ("contractnli", "cuad", "risknote"):
        rs = [r for r in recs if r["task"] == task]
        if not rs:
            continue
        exact = sum(norm(r["local"]) == norm(r["cluster"]) for r in rs)
        line = f"{task:12s} n={len(rs):2d}  identical output {exact}/{len(rs)}"
        if task != "risknote":
            lab = sum(r["local_label"] == r["cluster_label"] for r in rs)
            ev = sum(ev_key(r["local_ev"]) == ev_key(r["cluster_ev"]) for r in rs)
            line += f"  label {lab}/{len(rs)}  evidence {ev}/{len(rs)}"
        else:
            j = []
            for r in rs:
                a, b = set(norm(r["local"]).split()), set(norm(r["cluster"]).split())
                j.append(len(a & b) / len(a | b) if a | b else 1.0)
            line += f"  mean token Jaccard {sum(j) / len(j):.2f}"
        print(line)

    print("\n=== vs gold, 60-row smoke only (NOT project results) ===")
    for task, classes in (("contractnli", ["Entailment", "Contradiction", "NotMentioned"]),
                          ("cuad", [True, False])):
        rs = [r for r in recs if r["task"] == task]
        if not rs:
            continue
        for side in ("cluster", "local"):
            pairs = [(r["gold"], r[f"{side}_label"]) for r in rs]
            acc = sum(g == p for g, p in pairs) / len(pairs)
            print(f"{task:12s} {side:7s} acc {acc:.3f}  macro-F1 {macro_f1(pairs, classes):.3f}")

    flips = [r for r in recs if r["task"] != "risknote" and r["local_label"] != r["cluster_label"]]
    if flips:
        print("\nlabel flips:")
        for r in flips:
            print(f"  #{r['i']:2d} {r['task']:11s} gold={r['gold']} cluster={r['cluster_label']} "
                  f"local={r['local_label']} trunc={r['truncated']}")
    print("\nparse failures (local):",
          Counter(r["task"] for r in recs if r["task"] != "risknote" and r["local_label"] is None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "macbundle" / "validation_60.jsonl"))
    ap.add_argument("--model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--adapter", default=str(ROOT / "adapters" / "llama_3task_mlx"))
    # required: the file is appended to and resumed from, so two runs must not share it
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="local", choices=["local", "remote"])
    ap.add_argument("--endpoint", default=None, help="remote: base URL (default: .env LEGALAI_ENDPOINT)")
    ap.add_argument("--parallel", type=int, default=8, help="remote: concurrent requests")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task", default=None, help="only run one task")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    rows = [dict(r, i=i) for i, r in enumerate(rows)]
    if args.task:
        rows = [r for r in rows if r["task"] == args.task]
    rows = rows[: args.limit]

    eng = (RemoteEngine(endpoint=args.endpoint, parallel=args.parallel)
           if args.backend == "remote" else Engine(args.model, args.adapter))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if out_path.exists():  # resume: a 16k-token CUAD prefill is slow
        for l in open(out_path):
            rec = json.loads(l)
            done[rec["i"]] = rec

    with open(out_path, "a") as f:
        for r in rows:
            if r["i"] in done:
                continue
            t0 = time.time()
            res = eng.generate(r["messages"])
            ll, le = label_and_evidence(r["task"], res.text)
            cl, ce = label_and_evidence(r["task"], r["cluster_bf16_output"])
            rec = {"i": r["i"], "task": r["task"], "gold": gold_label(r),
                   "local": res.text, "cluster": r["cluster_bf16_output"],
                   "local_label": ll, "local_ev": le, "cluster_label": cl, "cluster_ev": ce,
                   "prompt_tokens": res.prompt_tokens, "truncated": res.truncated,
                   "secs": round(time.time() - t0, 1)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[r["i"]] = rec
            print(f"#{r['i']:2d} {r['task']:11s} {res.prompt_tokens:6d} tok "
                  f"{rec['secs']:6.1f}s  local={ll} cluster={cl}", flush=True)

    report([done[r["i"]] for r in rows])


if __name__ == "__main__":
    main()
