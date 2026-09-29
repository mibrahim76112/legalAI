#!/usr/bin/env python3
"""
Task 2 — CUAD clause identification.

Source: theatticusproject/cuad, file CUAD_v1/CUAD_v1.json, version aok_v1.0.
Official Atticus Project repository, CC BY 4.0, ungated. Not a mirror.

Target schema deliberately mirrors ContractNLI's:
    {"present": true/false, "evidence": ["..."]}

Split is BY CONTRACT with a recorded seed, so no contract has one clause in
train and another in test.

Evidence validation reuses reasoning_filters._grounded (which itself uses
text_norm.norm). No new normalizer, no new evidence metric.
"""
import os, sys, json, random, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                      # noqa: E402
from reasoning_filters import _grounded         # noqa: E402

SEED = 20260919
# Criterion: substantive, risk-bearing provisions only -- excludes the six
# metadata categories (present in 76-100% of contracts, near-degenerate as a
# binary target and unable to support a risk note), keeping the frequency band
# from Anti-Assignment (73.3%) down to Covenant Not To Sue (19.6%).
SELECTED = [
    "Anti-Assignment", "Cap On Liability", "License Grant", "Audit Rights",
    "Termination For Convenience", "Post-Termination Services", "Exclusivity",
    "Renewal Term", "Revenue/Profit Sharing", "Insurance", "Minimum Commitment",
    "Non-Transferable License", "Ip Ownership Assignment", "Change Of Control",
    "Non-Compete", "Notice Period To Terminate Renewal", "Uncapped Liability",
    "Covenant Not To Sue",
]
SYSTEM = (
    "You are a contract review assistant. Given the CONTRACT and a CLAUSE "
    "CATEGORY, determine whether the contract contains a clause of that "
    "category. If it does, return the exact sentence(s) from the contract that "
    "constitute it; if it does not, return an empty evidence list. Respond with "
    'JSON only, in the form {"present": true|false, "evidence": [...]}.'
)
USER_TMPL = "CONTRACT: {text}\n\nCLAUSE CATEGORY: {category}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v2/cuad")
    ap.add_argument("--dev-frac", type=float, default=0.12)
    ap.add_argument("--test-frac", type=float, default=0.20)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    src = hf_hub_download("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json",
                          repo_type="dataset")
    d = json.load(open(src, encoding="utf-8"))

    # ---- split BY CONTRACT, deterministic ----
    titles = sorted({e["title"] for e in d["data"]})
    rng = random.Random(SEED)
    rng.shuffle(titles)
    n = len(titles)
    n_test = int(round(args.test_frac * n)); n_dev = int(round(args.dev_frac * n))
    split_of = {}
    for t in titles[:n_test]:                    split_of[t] = "test"
    for t in titles[n_test:n_test + n_dev]:      split_of[t] = "dev"
    for t in titles[n_test + n_dev:]:            split_of[t] = "train"

    sel = set(SELECTED)
    rows = []
    checked = valid = invalid = dropped = 0
    per_cat = defaultdict(lambda: Counter())
    for e in d["data"]:
        sp = split_of[e["title"]]
        for par in e["paragraphs"]:
            ctx = par["context"]; hay = norm(ctx)
            for q in par["qas"]:
                cat = q["id"].split("__")[-1]
                if cat not in sel:
                    continue
                spans, bad = [], 0
                for a in q["answers"]:
                    t = (a.get("text") or "").strip()
                    if not t:
                        continue
                    checked += 1
                    if _grounded(t, hay):
                        valid += 1; spans.append(t)
                    else:
                        invalid += 1; bad += 1
                # a positive row whose every span failed validation has no
                # usable target -- drop it rather than emit present=true with []
                if q["answers"] and not spans:
                    dropped += 1
                    continue
                present = bool(spans)
                per_cat[cat][present] += 1
                rows.append({
                    "contract": e["title"], "category": cat, "split": sp,
                    "present": present, "evidence": spans,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER_TMPL.format(
                            text=ctx, category=cat)},
                        {"role": "assistant", "content": json.dumps(
                            {"present": present, "evidence": spans},
                            ensure_ascii=False)},
                    ]})

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for sp in ("train", "dev", "test"):
        sub = [r for r in rows if r["split"] == sp]
        with open(outd / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[sp] = len(sub)

    manifest = {
        "source": "theatticusproject/cuad :: CUAD_v1/CUAD_v1.json",
        "version": d.get("version"), "license": "CC BY 4.0",
        "official": True, "seed": SEED,
        "split_unit": "contract", "dev_frac": args.dev_frac,
        "test_frac": args.test_frac,
        "selection_criterion": ("substantive risk-bearing provisions only; the six "
                                "metadata categories (76-100% present) excluded as "
                                "near-degenerate binary targets that also cannot "
                                "support a risk note"),
        "selected_categories": SELECTED, "n_selected": len(SELECTED),
        "rows": dict(counts),
        "contracts": {sp: len({r["contract"] for r in rows if r["split"] == sp})
                      for sp in ("train", "dev", "test")},
        "evidence_validation": {"checked": checked, "valid": valid,
                                "invalid": invalid, "rows_dropped": dropped},
        "per_category": {c: {"present": per_cat[c][True],
                             "absent": per_cat[c][False]} for c in SELECTED},
    }
    json.dump(manifest, open(outd / "_manifest.json", "w"), indent=2)
    print(f"rows  {dict(counts)}   total {sum(counts.values())}")
    print(f"evidence: checked={checked} valid={valid} invalid={invalid} "
          f"rows_dropped={dropped}")
    print(f"-> {outd}/")


if __name__ == "__main__":
    main()
