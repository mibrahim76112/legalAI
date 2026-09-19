#!/usr/bin/env python3
"""Parse-quality inspector. Shows how outputs parsed and prints real failures,
so a formatting problem is never mistaken for a reasoning problem."""
import json, glob, argparse
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / "raw__*.jsonl")))
    if not files:
        print(f"no raw__*.jsonl in {args.dir}")
        return

    for fp in files:
        rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
        n = len(rows)
        modes = Counter(r["parse_mode"] for r in rows)
        jv = sum(r["json_valid"] for r in rows)
        fail = [r for r in rows if r["parsed_verdict"] is None]
        trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
        acc = sum(r["parsed_verdict"] == r["gold_verdict"] for r in rows) / n

        print(f"\n=== {Path(fp).name}  n={n} ===")
        print(f"  accuracy      : {acc:6.1%}")
        print(f"  json valid    : {jv/n:6.1%}")
        print(f"  parse failures: {len(fail)/n:6.1%}  ({len(fail)})")
        print(f"  truncated     : {trunc/n:6.1%}  ({trunc})")
        print(f"  parse modes   : {dict(modes)}")
        print(f"  predicted     : {dict(Counter(r['parsed_verdict'] for r in rows))}")
        for r in fail[:args.show]:
            print(f"    FAIL {r['example_id']}: {r['raw_output'][:180]!r}")


if __name__ == "__main__":
    main()
