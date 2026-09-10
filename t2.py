import json, os, zipfile
from collections import Counter
from dotenv import load_dotenv

# 1. Force Python to look for and read your .env file immediately
load_dotenv()

# 2. Extract the token now that variables are populated into your environment
hf_token = os.getenv("HF_TOKEN")

# 3. Pull in Hugging Face tools and explicitly log in 
from huggingface_hub import hf_hub_download, login

if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    print("CRITICAL: HF_TOKEN is still not detected by os.getenv(). Check your .env file placement.")

# ---------- 1. Load the FLATTENED file (what you'll train on) ----------
print("="*60)
print("FLATTENED FILE (kiddothe2b/contract-nli)")
print("="*60)

zip_path = hf_hub_download(repo_id="kiddothe2b/contract-nli", filename="contract_nli.zip", repo_type="dataset")
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("flat_data")

flat_rows = []
with open("flat_data/contract_nli_v1.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            flat_rows.append(json.loads(line))

flat_splits = {"train": [], "dev": [], "test": []}
for r in flat_rows:
    flat_splits[r["subset"]].append(r)

print("\nWhat one row looks like:")
print(json.dumps(flat_splits["train"][0], indent=2))

print("\nCounts:")
for split, data in flat_splits.items():
    print(f"  {split:5}: {len(data)} rows, labels = {dict(Counter(r['label'] for r in data))}")

# ---------- 2. Load the OFFICIAL file (the ground truth) ----------
print("\n" + "="*60)
print("OFFICIAL FILE (trismik/ContractNLI)")
print("="*60)

official = {}
for split in ("train", "dev", "test"):
    path = hf_hub_download(repo_id="trismik/ContractNLI", filename=f"{split}.json", repo_type="dataset")
    official[split] = json.load(open(path, encoding="utf-8"))

print("\nWhat one document looks like (truncated text):")
sample_doc = official["train"]["documents"][0]
preview = dict(sample_doc)
preview["text"] = preview["text"][:200] + "..."
preview["spans"] = preview["spans"][:3] + ["...more..."]
print(json.dumps(preview, indent=2))

print("\nWhat the hypothesis labels look like:")
print(json.dumps(dict(list(official["train"]["labels"].items())[:2]), indent=2))

# ---------- 3. Completeness check: does what you have match the paper? ----------
print("\n" + "="*60)
print("COMPLETENESS CHECK")
print("="*60)

expected_docs = {"train": 423, "dev": 61, "test": 123}
n_hypotheses = len(official["train"]["labels"])

for split in ("train", "dev", "test"):
    n_docs = len(official[split]["documents"])
    n_official_pairs = sum(
        len(ann["annotations"]) for doc in official[split]["documents"] for ann in doc["annotation_sets"]
    )
    n_flat_pairs = len(flat_splits[split])
    expected_pairs = n_docs * n_hypotheses

    ok_docs = "OK" if n_docs == expected_docs[split] else "MISMATCH"
    print(f"\n{split}:")
    print(f"  documents          : {n_docs} (expected {expected_docs[split]}) -> {ok_docs}")
    print(f"  official pairs     : {n_official_pairs} (docs x {n_hypotheses} hyps = {expected_pairs})")
    print(f"  flattened pairs    : {n_flat_pairs}")
    print(f"  flattened coverage : {n_flat_pairs/n_official_pairs*100:.1f}% of official")