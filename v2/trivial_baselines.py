#!/usr/bin/env python3
"""Trivial (no-model) baselines. Distinct from the zero-shot baseline.

  TRIVIAL baseline  = no model at all. Always predict the majority class,
                      always return empty evidence. Answers "how much of this
                      score is just the label distribution?"
  ZERO-SHOT baseline = the real base model with no adapter. Answers "how much
                      did fine-tuning add?"

Both belong in the deck; they answer different questions and neither
substitutes for the other.
"""
import sys, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1_contractnli"))
from doc_harness import CLASSES                                # noqa: E402

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)

def macro(pred, gold, classes):
    fs=[]
    for c in classes:
        tp=sum(1 for a,b in zip(pred,gold) if a==c and b==c)
        fp=sum(1 for a,b in zip(pred,gold) if a==c and b!=c)
        fn=sum(1 for a,b in zip(pred,gold) if a!=c and b==c)
        fs.append(prf(tp,fp,fn)[2])
    return sum(fs)/len(fs)

out={}
for split in ("dev","test"):
    rows=[json.loads(l) for l in open(f"v2/combined/{split}.jsonl")]
    cn=[r for r in rows if r["task"]=="contractnli"]
    cu=[r for r in rows if r["task"]=="cuad"]
    g1=[r["verdict"] for r in cn]
    dist=Counter(g1); maj=dist.most_common(1)[0][0]
    b1={}
    for name,pred in (("majority_class_"+maj,[maj]*len(g1)),
                      ("always_Contradiction",["Contradiction"]*len(g1)),
                      ("always_NotMentioned",["NotMentioned"]*len(g1))):
        acc=sum(1 for a,b in zip(pred,g1) if a==b)/len(g1)
        b1[name]={"accuracy":acc,"macro_f1":macro(pred,g1,CLASSES)}
    b1["label_distribution"]={k:v/len(g1) for k,v in dist.items()}
    b1["n"]=len(g1)

    g2=[bool(r["present"]) for r in cu]
    npres=sum(g2); n=len(g2)
    b2={"n":n,"present_rate":npres/n,
        "constant_ABSENT":{"accuracy":(n-npres)/n,"present_precision":0.0,
                           "present_recall":0.0,"present_f1":0.0,
                           "macro_f1_present":0.0},
        "constant_PRESENT":{"accuracy":npres/n,"present_precision":npres/n,
                            "present_recall":1.0,
                            "present_f1":2*(npres/n)/((npres/n)+1)}}
    # evidence: the "return nothing" baseline
    ev1=[r for r in cn if r.get("evidence")]
    ev2=[r for r in cu if r.get("evidence")]
    b1["evidence_empty_predictor"]={
        "gold_bearing":len(ev1),"gold_empty":len(cn)-len(ev1),
        "strict_TP":0,"strict_FN":len(ev1),"strict_FP":0,
        "strict_TN":len(cn)-len(ev1),"strict_f1":0.0,
        "note":"returns [] always: every gold-bearing row is a FN, every "
               "gold-empty row a TN, so strict F1 is exactly 0"}
    b2["evidence_empty_predictor"]={
        "gold_bearing":len(ev2),"gold_empty":len(cu)-len(ev2),"strict_f1":0.0}
    out[split]={"task1_contractnli":b1,"task2_cuad":b2}

json.dump(out,open("v2/_trivial_baselines.json","w"),indent=2)
for split in ("dev","test"):
    b=out[split]
    print(f"\n=== {split.upper()} ===")
    a=b["task1_contractnli"]
    print(f"ContractNLI (n={a['n']}) label dist: "
          + ", ".join(f"{k} {v:.1%}" for k,v in a['label_distribution'].items()))
    for k,v in a.items():
        if isinstance(v,dict) and "accuracy" in v:
            print(f"  {k:28s} acc {v['accuracy']:.3f}  macroF1 {v['macro_f1']:.3f}")
    c=b["task2_cuad"]
    print(f"CUAD (n={c['n']}) present rate {c['present_rate']:.1%}")
    print(f"  constant_ABSENT              acc {c['constant_ABSENT']['accuracy']:.3f}  "
          f"present-F1 {c['constant_ABSENT']['present_f1']:.3f}")
    print(f"  constant_PRESENT             acc {c['constant_PRESENT']['accuracy']:.3f}  "
          f"present-F1 {c['constant_PRESENT']['present_f1']:.3f}")
print("\n-> v2/_trivial_baselines.json")
