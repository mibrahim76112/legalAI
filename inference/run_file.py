"""Review one contract file from the command line, without the server.

    venv-mlx/bin/python inference/run_file.py ~/Downloads/test_nda.pdf --out results/test_nda_review.json
"""

import argparse
import json
import time
from pathlib import Path

from engine import Engine
from extract import extract_text
from pipeline import Reviewer
from server import ADAPTER, MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="compliance,clauses")
    args = ap.parse_args()

    path = Path(args.file).expanduser()
    text = extract_text(path.name, path.read_bytes())
    t0 = time.time()

    def progress(**kw):
        print(f"[{time.time() - t0:6.1f}s] {kw}", flush=True)

    review = Reviewer(Engine(MODEL, ADAPTER)).run(
        text, tasks=args.tasks.split(","), document_name=path.stem, progress=progress)
    Path(args.out).write_text(json.dumps(review, indent=1))
    for it in review["compliance"] + review["clauses"]:
        print(f"{it['status']:16s} {len(it['evidence'])} ev  {it['title']}")
    print("stats", review["stats"], f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
