#!/usr/bin/env python3
"""Unit tests for document-level parsing + evidence matching.
A matcher bug looks exactly like a bad model, so it gets tested first."""
from doc_harness import parse_output, evidence_case, covered_flags, jaccard, norm

P_CASES = [
    # (raw, verdict, evidence, json_state, schema_valid)
    ('{"verdict": "Entailment", "evidence": ["A b."]}', "Entailment", ["A b."], "json", True),
    ('{"verdict":"NotMentioned","evidence":[]}', "NotMentioned", [], "json", True),
    ('```json\n{"verdict": "Contradiction", "evidence": ["X"]}\n```',
     "Contradiction", ["X"], "json", True),
    ('Sure! {"verdict": "Entailment", "evidence": ["Y"]} done',
     "Entailment", ["Y"], "json", True),
    # thinking cells generate a real block; it must not decide the answer
    ('<think>maybe Contradiction</think>{"verdict":"Entailment","evidence":[]}',
     "Entailment", [], "json", True),
    # truncated thinking block, no closing tag content after
    ('<think>deliberating forever', None, [], "not_json", False),
    # Gemma channel syntax
    ('<|channel>thought\nhmm<channel|><|channel>final\n{"verdict":"Entailment","evidence":["Z"]}<channel|>',
     "Entailment", ["Z"], "json", True),
    # valid JSON, wrong key -> must be json_wrong_schema, NOT not_json
    ('{"Relation": "Entailment"}', "Entailment", [], "json_wrong_schema", False),
    # evidence given as a bare string rather than a list
    ('{"verdict":"Entailment","evidence":"single"}', "Entailment", ["single"], "json", True),
    # bare label only
    ('Entailment', "Entailment", [], "not_json", False),
    ('', None, [], "not_json", False),
    ('{"verdict":"Maybe","evidence":[]}', None, [], "json_wrong_schema", False),
]

E_CASES = [
    # (gold, pred, expected_case, expected_frac)
    (["Hello  World."], ["hello world."], "TP", 1.0),        # normalization
    (["A.", "B."], ["A. B."], "TP", 1.0),                     # concatenated hay
    (["A.", "B."], ["A."], "FN", 0.5),                        # partial
    (["A."], [], "FN", 0.0),                                  # empty pred, gold non-empty
    ([], ["anything"], "FP", 1.0),                            # hallucinated evidence
    ([], [], "TN", 1.0),                                      # correct abstention
    (["The  quick\nbrown fox"], ["the quick brown fox"], "TP", 1.0),
]


def main():
    bad = 0
    for raw, v, ev, js, sv in P_CASES:
        r = parse_output(raw)
        if (r["verdict"], r["evidence"], r["json_state"], r["schema_valid"]) != (v, ev, js, sv):
            bad += 1
            print(f"PARSE FAIL {raw[:52]!r}\n   got {(r['verdict'], r['evidence'], r['json_state'], r['schema_valid'])}"
                  f"\n  want {(v, ev, js, sv)}")
    for gold, pred, case, frac in E_CASES:
        c, f = evidence_case(gold, pred)
        if (c, round(f, 3)) != (case, round(frac, 3)):
            bad += 1
            print(f"EVID FAIL gold={gold} pred={pred} got {(c, f)} want {(case, frac)}")
    # jaccard sanity
    assert jaccard(["a b"], ["a b"]) == 1.0
    assert jaccard(["a b"], ["a c"]) == 1 / 3
    assert jaccard([], []) is None
    n = len(P_CASES) + len(E_CASES)
    print(f"{n - bad}/{n} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
