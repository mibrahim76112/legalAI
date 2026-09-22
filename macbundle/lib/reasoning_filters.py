#!/usr/bin/env python3
"""
Arm B filters. Pure string checks, no LLM judge (per the brief).

Reuses doc_harness.norm -- the SAME normalizer the Arm A evaluation uses.
No second normalizer exists anywhere in this project.
"""
import re, json, statistics as st
from collections import Counter
from doc_harness import norm

# quoted spans inside the reasoning prose: "..." or '...'
_Q = re.compile(r'"([^"]{8,})"' r"|'([^']{8,})'")
CLASSES = ["Entailment", "Contradiction", "NotMentioned"]


def quoted_spans(reasoning):
    out = []
    for a, b in _Q.findall(reasoning or ""):
        s = (a or b).strip()
        if s:
            out.append(s)
    return out


def _grounded(q, hay):
    """Substring test, tolerant of edge punctuation the teacher adds when it
    quotes a clause HEADING: the contract has "KEEP INFORMATION CONFIDENTIAL"
    and the teacher writes "KEEP INFORMATION CONFIDENTIAL." Trailing/leading
    punctuation is the model's sentence-building, not a fabricated quote.
    Still a pure string check -- no fuzzy matching, no edit distance.
    """
    n = norm(q)
    if n in hay:
        return True
    return n.strip(".,;:!?\u2019'\"-\u2014 ") in hay


def f1_quotes_grounded(row, contract):
    """Every quoted string in the reasoning must occur in the contract."""
    hay = norm(contract)
    bad = [q for q in quoted_spans(row["reasoning"]) if not _grounded(q, hay)]
    return (not bad), bad


def f2_evidence_matches_gold(row):
    """Arm A's rule: gold non-empty AND every gold span is a contiguous
    substring of the concatenated normalized prediction.

    Also classifies HOW a mismatch fails, which is the interesting part:
      superset  - every gold span covered AND the teacher added more
      partial   - some gold spans covered, some not
      disjoint  - no gold span covered at all
      spurious  - gold empty, teacher supplied evidence
      abstain   - gold non-empty, teacher supplied nothing
    """
    gold, pred = row["gold_evidence"], row["evidence"]
    hay = norm(" ".join(pred))
    if not gold:
        return (not pred), ("ok" if not pred else "spurious")
    if not pred:
        return False, "abstain"
    cov = [norm(g) in hay for g in gold if norm(g)]
    if all(cov):
        # covered; is the teacher also carrying text beyond gold?
        gold_tok = set(norm(" ".join(gold)).split())
        pred_tok = set(hay.split())
        return (True, "exact" if len(pred_tok - gold_tok) <= 5 else "superset")
    return False, ("disjoint" if not any(cov) else "partial")


# The teacher asserts its verdict in natural prose ("the position is not
# mentioned in the contract"), not as the schema token. Matching the literal
# class name rejected 29.7% of rows for a reason that was purely my own.
# Stems, not inflected forms. "contradicts" is NOT a substring of
# "contradicted", and the teacher overwhelmingly writes the passive
# ("the position is contradicted by the contract") -- 226 rows were rejected for
# that alone, which depleted the Contradiction class specifically.
_VERDICT_STEMS = {
    "NotMentioned": ["notmentioned", "not mentioned", "does not address",
                     "not addressed", "no clause addresses", "silent on",
                     "does not mention", "is not addressed"],
    "Entailment": ["entail", "supports the position", "supports the stated",
                   "is supported by", "consistent with the position"],
    "Contradiction": ["contradict", "conflicts with", "in conflict with",
                      "contrary to", "is inconsistent with", "prohibits the position",
                      "does not permit"],
}
# Negated forms must NOT count as an assertion of that class.
_NEG = re.compile(r"(does not|do not|is not|are not|no|nor|without)\s+(\w+\s+){0,2}$")


def _asserts(txt, cls):
    """Is class `cls` asserted (not negated) anywhere in the prose? Returns the
    last character offset of an un-negated assertion, or -1."""
    best = -1
    for p in _VERDICT_STEMS[cls]:
        start = 0
        pn = norm(p)
        while True:
            i = txt.find(pn, start)
            if i < 0:
                break
            if not _NEG.search(txt[max(0, i - 40):i]):
                best = max(best, i)
            start = i + 1
    return best


def f3_verdict_consistent(row):
    """The verdict ASSERTED in the prose must equal the JSON verdict field."""
    v = row.get("verdict")
    if not v:
        return False, "no_verdict"
    txt = norm(row["reasoning"])
    pos = {c: _asserts(txt, c) for c in CLASSES}
    named = [c for c, i in pos.items() if i >= 0]
    if not named:
        return False, "verdict_not_stated_in_prose"
    # the CONCLUSION is the class asserted last, not merely mentioned
    last = max(named, key=lambda c: pos[c])
    return (last == v), ("ok" if last == v else f"prose_concludes_{last}")


def derive_length_bounds(rows, hard_min_tokens=20):
    """p1/p99 of the observed distribution, with a hard floor: a rationale under
    ~20 tokens cannot contain a clause name plus a verbatim quote plus an
    application. Derived from data, reported, not hardcoded blind."""
    t = sorted(r["n_output_tokens"] for r in rows if r.get("reasoning"))
    if not t:
        return hard_min_tokens, 10**9
    p1 = t[max(0, int(0.01 * len(t)) - 1)]
    p99 = t[min(len(t) - 1, int(0.99 * len(t)))]
    return max(p1, hard_min_tokens), p99


def f4_length_ok(row, lo, hi):
    n = row["n_output_tokens"]
    if row.get("finish_reason") == "length":
        return False, "truncated_at_cap"
    if n < lo:
        return False, f"too_short(<{lo})"
    if n > hi:
        return False, f"too_long(>{hi})"
    return True, "ok"


# Coherence requires the reasoning not CONTRADICT the gold evidence, not that it
# enumerate it exhaustively. Accepting partial/superset alongside exact recovers
# rows whose verdict is right and whose cited clause overlaps gold, and rejects
# only the two incoherent modes: disjoint (entirely different text) and spurious
# (gold empty, teacher cited anyway). Ablatable via f2_strict=True.
F2_ACCEPT_RELAXED = {"ok", "exact", "superset", "partial"}
F2_ACCEPT_STRICT = {"ok", "exact"}


def apply_all(rows, contracts, lo, hi, f2_strict=False):
    kept, rejected = [], []
    reasons = Counter()
    ev_modes = Counter()
    for r in rows:
        why = []
        ok1, bad = f1_quotes_grounded(r, contracts.get(r["doc_id"], ""))
        if not ok1:
            why.append(f"f1_ungrounded_quote({len(bad)})")
        _, mode = f2_evidence_matches_gold(r)
        ev_modes[mode] += 1
        accept = F2_ACCEPT_STRICT if f2_strict else F2_ACCEPT_RELAXED
        if mode not in accept:
            why.append(f"f2_evidence_{mode}")
        ok3, m3 = f3_verdict_consistent(r)
        if not ok3:
            why.append(f"f3_{m3}")
        ok4, m4 = f4_length_ok(r, lo, hi)
        if not ok4:
            why.append(f"f4_{m4}")
        if why:
            reasons[why[0]] += 1          # first failure, for a clean breakdown
            rejected.append({**r, "reject_reasons": why})
        else:
            kept.append(r)
    return kept, rejected, reasons, ev_modes
