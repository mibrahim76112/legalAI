#!/usr/bin/env python3
"""Can retrieval alone find the evidence the fine-tuned model quotes?

Answers the obvious reviewer question: why fine-tune instead of just doing RAG?

Setup. Each contract is chunked into overlapping passages and indexed twice:
  - BM25, lexical. A strong baseline on legal text, where exact terminology
    ("indemnify", "assign", "terminate for convenience") carries the signal.
  - Dense, BAAI/bge-base-en-v1.5 cosine similarity over normalized embeddings.
  - Hybrid, Reciprocal Rank Fusion of the two. RRF operates on RANKS, not
    scores, because BM25 is unbounded and cosine sits in [-1,1]; averaging
    them directly is meaningless without per-corpus calibration.
        RRF(d) = sum over retrievers of 1 / (k_rrf + rank(d))
  - Hybrid + rerank, a BAAI/bge-reranker-base cross-encoder applied to the top
    RERANK_N fused candidates. A cross-encoder reads query and passage jointly
    rather than embedding them separately, which is why it reranks well and
    why it is too slow to use as a first-stage retriever.

Query is the thing the reviewer actually asks:
  ContractNLI -> the playbook position sentence
  CUAD        -> the clause category name

Metric is RECALL@k: for a gold-bearing example, is EVERY gold span contained in
the union of the top-k retrieved chunks? That is deliberately the same
all-or-nothing rule as the strict evidence metric, so the number is directly
comparable to the fine-tuned model's evidence recall rather than a friendlier
variant. A looser any-span variant is reported alongside.

This measures the CEILING retrieval imposes. A generator downstream can only
quote what retrieval hands it, so recall@k upper-bounds any RAG pipeline built
this way.
"""
import json, re, sys, math, argparse
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from text_norm import norm                                     # noqa: E402
from reasoning_filters import _grounded                        # noqa: E402

WORD = re.compile(r"[a-z0-9]+")


def parent_child(text, child=300, parent=1200, overlap=80):
    """Index small children for precise matching; return the enclosing parent.

    Targets the measured failure mode directly: a gold span that straddles a
    chunk boundary is unreachable at any k with flat chunking, because no
    single chunk contains all of it. A small child can still MATCH the query
    while the parent it expands to carries the whole clause.
    """
    kids, i = [], 0
    while i < len(text):
        j = min(i + child, len(text))
        if j < len(text):
            sp = text.rfind(" ", i + child // 2, j)
            if sp > i:
                j = sp
        mid = (i + j) // 2
        lo = max(0, mid - parent // 2)
        hi = min(len(text), lo + parent)
        kids.append({"child": text[i:j], "parent": text[lo:hi]})
        if j >= len(text):
            break
        i = max(i + 1, j - overlap)
    return kids


def chunk(text, size=900, overlap=200):
    """Character windows with overlap, snapped to whitespace so clauses aren't
    cut mid-word. Overlap matters: a gold span straddling a boundary would
    otherwise be unreachable by any k."""
    out, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        if j < len(text):
            sp = text.rfind(" ", i + size // 2, j)
            if sp > i:
                j = sp
        out.append((i, text[i:j]))
        if j >= len(text):
            break
        i = max(i + 1, j - overlap)
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [WORD.findall(d.lower()) for d in docs]
        self.len = [len(d) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        self.df = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.N = len(self.docs)
        self.tf = [Counter(d) for d in self.docs]

    def score(self, q):
        qs = WORD.findall(q.lower())
        out = [0.0] * self.N
        for t in qs:
            if t not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if not f:
                    continue
                out[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--chunk", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rerank-n", type=int, default=20,
                    help="how many fused candidates the cross-encoder sees")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--parent-child", action="store_true",
                    help="index 300-char children, return 1200-char parents")
    ap.add_argument("--child", type=int, default=300)
    ap.add_argument("--parent", type=int, default=1200)
    ap.add_argument("--out", default="v2/_rag_retrieval.json")
    args = ap.parse_args()

    dev = [json.loads(l) for l in open("v2/combined/dev.jsonl", encoding="utf-8")]

    def user(r):
        return [m for m in r["messages"] if m["role"] == "user"][0]["content"]

    # group queries by contract, recovering the document text once per contract
    docs, queries = {}, []
    for r in dev:
        if r["task"] == "contractnli":
            m = re.search(r"^CONTRACT:\s*(.*?)\s*\n\nQUESTION:\s*(.*)$", user(r), re.S)
            if not m:
                continue
            key = ("nli", r["doc_id"])
            docs.setdefault(key, m.group(1))
            queries.append({"task": "contractnli", "key": key,
                            "query": m.group(2).strip(),
                            "gold": r.get("evidence") or []})
        elif r["task"] == "cuad":
            tail = f"\n\nCLAUSE CATEGORY: {r['category']}"
            key = ("cuad", r["contract"])
            docs.setdefault(key, user(r)[len("CONTRACT: "):-len(tail)])
            queries.append({"task": "cuad", "key": key,
                            "query": r["category"],
                            "gold": r.get("evidence") or []})
    gold_q = [q for q in queries if q["gold"]]
    print(f"[data] {len(docs)} contracts, {len(queries)} queries, "
          f"{len(gold_q)} with gold evidence", flush=True)

    from sentence_transformers import SentenceTransformer
    import numpy as np
    enc = SentenceTransformer(args.model, device="cpu")

    index = {}
    for key, text in docs.items():
        if args.parent_child:
            kids = parent_child(text, args.child, args.parent, args.overlap // 2)
            idx_texts = [k["child"] for k in kids]      # what we MATCH on
            ret_texts = [k["parent"] for k in kids]     # what we RETURN
        else:
            ch = chunk(text, args.chunk, args.overlap)
            idx_texts = ret_texts = [c[1] for c in ch]
        emb = enc.encode(idx_texts, batch_size=32, normalize_embeddings=True,
                         show_progress_bar=False)
        index[key] = {"chunks": ret_texts, "match": idx_texts,
                      "bm25": BM25(idx_texts), "emb": np.asarray(emb)}
    nch = sum(len(v["chunks"]) for v in index.values())
    mode = (f"parent-child child={args.child} parent={args.parent}"
            if args.parent_child else f"flat size={args.chunk} overlap={args.overlap}")
    print(f"[index] {nch} units, {nch/len(index):.0f} per contract  [{mode}]", flush=True)

    METHODS = ["bm25", "dense", "rrf"] + ([] if args.no_rerank else ["rrf_rerank"])
    res = {m: {t: {k: {"all": 0, "any": 0, "n": 0} for k in args.ks}
               for t in ("contractnli", "cuad")} for m in METHODS}
    ce = None
    if not args.no_rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(args.reranker, device="cpu", max_length=512)
        print(f"[rerank] {args.reranker} over top-{args.rerank_n} fused", flush=True)
    qtexts = [q["query"] for q in gold_q]
    qemb = enc.encode(qtexts, batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False)
    for qi, q in enumerate(gold_q):
        I = index[q["key"]]
        order = {}
        s = I["bm25"].score(q["query"])
        order["bm25"] = sorted(range(len(s)), key=lambda i: -s[i])
        sim = I["emb"] @ qemb[qi]
        order["dense"] = list(np.argsort(-sim))
        # --- RRF: rank-based fusion, no score calibration needed ---
        rr = defaultdict(float)
        for m in ("bm25", "dense"):
            for rank, idx in enumerate(order[m]):
                rr[idx] += 1.0 / (args.rrf_k + rank + 1)
        order["rrf"] = sorted(rr, key=lambda i: -rr[i])
        # --- cross-encoder rerank of the fused head ---
        if ce is not None:
            cand = order["rrf"][:args.rerank_n]
            pairs = [(q["query"], I["chunks"][i][:1500]) for i in cand]
            sc = ce.predict(pairs, batch_size=32, show_progress_bar=False)
            order["rrf_rerank"] = [c for _, c in
                                   sorted(zip(sc, cand), key=lambda x: -x[0])]
        for m in METHODS:
            for k in args.ks:
                hay = norm(" ".join(I["chunks"][i] for i in order[m][:k]))
                hits = [g for g in q["gold"] if _grounded(g, hay)]
                cell = res[m][q["task"]][k]
                cell["n"] += 1
                cell["all"] += int(len(hits) == len(q["gold"]))
                cell["any"] += int(bool(hits))

    print()
    for t, lab in (("contractnli", "ContractNLI"), ("cuad", "CUAD")):
        n = res["bm25"][t][args.ks[0]]["n"]
        print(f"{lab}  (n={n} gold-bearing queries)")
        print("  " + f"{'k':>3s}" + "".join(f"{m:>13s}" for m in METHODS)
              + "      (recall@k, ALL gold spans retrieved)")
        for k in args.ks:
            row = "".join(f"{res[m][t][k]['all']/res[m][t][k]['n']*100:12.1f}%"
                          for m in METHODS)
            print(f"  {k:>3d}" + row)
        print()
    json.dump({"model": args.model, "chunk": args.chunk, "overlap": args.overlap,
               "parent_child": args.parent_child, "child": args.child,
               "parent": args.parent, "reranker": None if args.no_rerank else args.reranker,
               "n_contracts": len(docs), "n_chunks": nch, "ks": args.ks,
               "results": res}, open(args.out, "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
