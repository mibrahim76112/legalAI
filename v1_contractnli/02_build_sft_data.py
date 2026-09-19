"""Build mlx-lm LoRA training files from the flattened ContractNLI corpus.

flat_data/contract_nli_v1.jsonl -> sft_data/<variant>/{train,valid,test}.jsonl
"""

import argparse
import collections
import hashlib
import json
import os
import random

SRC = "flat_data/contract_nli_v1.jsonl"
OUT_ROOT = "sft_data"

# flattened corpus uses lowercase names; official release uses these
LABEL_MAP = {
    "entailment": "Entailment",
    "contradiction": "Contradiction",
    "neutral": "NotMentioned",
}

# guard only; shortest real premise is 33 chars
MIN_PREMISE_CHARS = 20

# mlx-lm requires these exact file names
SUBSET_TO_FILE = {"train": "train", "dev": "valid", "test": "test"}

# withheld from training; leaves the main contradiction sources in train
HELDOUT_HYPOTHESES = ["nda-1", "nda-8", "nda-11", "nda-13"]

SYSTEM_PROMPT = (
    "You are reviewing a clause from a Non-Disclosure Agreement against a "
    "company policy position. Given the CLAUSE and the POLICY, determine "
    "whether the clause supports the policy (Entailment), conflicts with it "
    "(Contradiction), or does not address it (NotMentioned). Answer with a "
    "single JSON object and nothing else."
)

USER_TEMPLATE = "CLAUSE: {premise}\nPOLICY: {hypothesis}"


def hypothesis_ids():
    """Map hypothesis text -> nda-N id."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="trismik/ContractNLI", filename="dev.json", repo_type="dataset"
    )
    with open(path, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    return {v["hypothesis"]: k for k, v in labels.items()}


def load_rows():
    rows = []
    dropped = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            premise = r["premise"].strip()
            if len(premise) < MIN_PREMISE_CHARS:
                dropped += 1
                continue
            rows.append(
                {
                    "premise": premise,
                    "hypothesis": r["hypothesis"].strip(),
                    "label": LABEL_MAP[r["label"]],
                    "subset": r["subset"],
                }
            )
    return rows, dropped


def chunk_id(premise):
    """Stable id for a premise chunk; ignored by mlx-lm, used for citation."""
    return hashlib.sha1(premise.encode("utf-8")).hexdigest()[:12]


def to_example(row):
    return {
        "chunk_id": chunk_id(row["premise"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(**row)},
            {"role": "assistant", "content": json.dumps({"verdict": row["label"]})},
        ]
    }


def write_split(out_dir, name, rows):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_example(r), ensure_ascii=False) + "\n")
    counts = collections.Counter(r["label"] for r in rows)
    print(f"  {path:44} {len(rows):5} rows  {dict(counts)}")


def oversample(rows, target_label="Contradiction", seed=0):
    """Duplicate minority rows up to the largest class."""
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    biggest = max(len(v) for v in by_label.values())
    minority = by_label[target_label]
    if not minority:
        return list(rows)
    rng = random.Random(seed)
    extra = [rng.choice(minority) for _ in range(biggest - len(minority))]
    out = list(rows) + extra
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["baseline", "oversampled", "heldout_hypothesis", "all"],
        default="all",
    )
    args = ap.parse_args()

    rows, dropped = load_rows()
    print(f"loaded {len(rows)} rows from {SRC} (dropped {dropped} under "
          f"{MIN_PREMISE_CHARS} chars)")

    by_subset = collections.defaultdict(list)
    for r in rows:
        by_subset[r["subset"]].append(r)

    want = ["baseline", "oversampled", "heldout_hypothesis"]
    if args.variant != "all":
        want = [args.variant]

    if "baseline" in want:
        print("\nbaseline")
        out = os.path.join(OUT_ROOT, "baseline")
        for subset, fname in SUBSET_TO_FILE.items():
            write_split(out, fname, by_subset[subset])

    if "oversampled" in want:
        print("\noversampled (train only)")
        out = os.path.join(OUT_ROOT, "oversampled")
        write_split(out, "train", oversample(by_subset["train"]))
        write_split(out, "valid", by_subset["dev"])
        write_split(out, "test", by_subset["test"])

    if "heldout_hypothesis" in want:
        print(f"\nheldout_hypothesis (unseen: {HELDOUT_HYPOTHESES})")
        hyp2id = hypothesis_ids()
        held = set(HELDOUT_HYPOTHESES)
        seen_rows, unseen_rows = [], []
        for r in rows:
            (unseen_rows if hyp2id[r["hypothesis"]] in held else seen_rows).append(r)
        seen_by = collections.defaultdict(list)
        for r in seen_rows:
            seen_by[r["subset"]].append(r)
        out = os.path.join(OUT_ROOT, "heldout_hypothesis")
        write_split(out, "train", seen_by["train"])
        write_split(out, "valid", seen_by["dev"])
        write_split(out, "test", unseen_rows)


if __name__ == "__main__":
    main()
