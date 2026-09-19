#!/usr/bin/env python3
"""Parser unit tests. Run before burning GPU hours: a parser bug looks
exactly like a bad model, and that is the failure mode we most want to avoid."""
from harness_core import parse_verdict as P

CASES = [
    # (raw, expected_verdict, expected_json_valid)
    ('{"verdict": "Entailment"}',                      "Entailment",    True),
    ('{"verdict":"Contradiction"}',                    "Contradiction", True),
    ('{"verdict": "NotMentioned"}',                    "NotMentioned",  True),
    ('  {"verdict": "Entailment"}  ',                  "Entailment",    True),
    ('```json\n{"verdict": "Entailment"}\n```',        "Entailment",    True),
    ('Here is my answer: {"verdict": "Contradiction"}', "Contradiction", True),
    ('{"verdict": "not mentioned"}',                   "NotMentioned",  True),
    ('{"verdict": "NOT_MENTIONED"}',                   "NotMentioned",  True),
    ('{"verdict": "neutral"}',                         "NotMentioned",  True),
    ('{"reason": "x", "verdict": "Entailment"}',       "Entailment",    True),
    # reasoning-model wrappers: the think block must not decide the answer
    ('<think>Maybe Contradiction...</think>{"verdict": "Entailment"}',
                                                        "Entailment",   True),
    ('<think>long deliberation</think>\nEntailment',   "Entailment",    False),
    # bare labels parse, but are NOT json_valid
    ('Entailment',                                      "Entailment",   False),
    ('The verdict is contradiction.',                   "Contradiction",False),
    # genuine failures
    ('',                                                None,           False),
    (None,                                              None,           False),
    ('I cannot determine this.',                        None,           False),
    ('{"verdict": "Maybe"}',                            None,           False),
]

def main():
    bad = 0
    for raw, want_v, want_j in CASES:
        v, j, how = P(raw)
        ok = (v == want_v and j == want_j)
        if not ok:
            bad += 1
            print(f"FAIL {raw!r:55.55} -> ({v}, {j}, {how})  want ({want_v}, {want_j})")
    print(f"{len(CASES)-bad}/{len(CASES)} passed")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
