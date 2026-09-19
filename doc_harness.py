#!/usr/bin/env python3
"""Document-level prompts, parsing, and evidence matching. Shared by all cells."""
import json, re

CLASSES = ["Entailment", "Contradiction", "NotMentioned"]

SYSTEM = (
    "You are a junior legal assistant reviewing a contract. Given the CONTRACT "
    "and the QUESTION, determine whether the contract supports the stated position "
    "(Entailment), conflicts with it (Contradiction), or does not address it "
    "(NotMentioned). Return the exact sentence(s) from the contract that justify "
    "your answer, or an empty evidence list if none apply. Respond with JSON only, "
    "in the form {\"verdict\": \"...\", \"evidence\": [...]}."
)
USER_TMPL = "CONTRACT: {text}\n\nQUESTION: {hypothesis}"


def build_messages(text, hypothesis):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(text=text, hypothesis=hypothesis)}]


# ---- normalization ----------------------------------------------------------
# Single shared definition; see text_norm.py for why it is not defined here.
from text_norm import norm  # noqa: F401  (re-exported: many modules import it here)


# ---- output cleaning ----------------------------------------------------
# Reasoning wrappers belong to the PROMPT in non-thinking mode, but thinking
# cells generate them, and Gemma 4 speaks in channels. Strip both before
# parsing so the JSON is found in the answer, not the deliberation.
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.S | re.I)
_CHANNEL_THOUGHT_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.S)
_CHANNEL_ANY_RE = re.compile(r"<\|channel>\w+\s*|<channel\|>|<\|turn>\w*\s*|<turn\|>")
_FENCE_RE = re.compile(r"```(?:json)?|```", re.I)


def strip_wrappers(raw):
    t = raw or ""
    t = _CHANNEL_THOUGHT_RE.sub(" ", t)
    t = _THINK_RE.sub(" ", t)
    if "</think>" in t.lower():          # unclosed/truncated thinking block
        t = _OPEN_THINK_RE.sub(" ", t)
    t = _CHANNEL_ANY_RE.sub(" ", t)
    t = _FENCE_RE.sub(" ", t)
    return t.strip()


def _find_json(text):
    try:
        o = json.loads(text)
        if isinstance(o, dict):
            return o
    except Exception:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    o = json.loads(text[start:i + 1])
                    if isinstance(o, dict):
                        return o
                except Exception:
                    start = None
    return None


_VERDICT_RE = re.compile(r"\b(entailment|contradiction|not[\s_-]?mentioned)\b", re.I)
_CANON = {"entailment": "Entailment", "contradiction": "Contradiction",
          "notmentioned": "NotMentioned"}


def _canon_verdict(v):
    if not isinstance(v, str):
        return None
    return _CANON.get(re.sub(r"[\s_-]", "", v).strip().lower())


def parse_output(raw):
    """-> dict with verdict, evidence, and format diagnostics.

    json_state is three-way on purpose; conflating "not JSON" with
    "JSON, wrong schema" has misled us before.
    """
    text = strip_wrappers(raw)
    out = {"verdict": None, "evidence": [], "json_state": "not_json",
           "schema_valid": False, "parse_mode": "unparseable"}
    if not text:
        out["parse_mode"] = "empty"
        return out

    obj = _find_json(text)
    if obj is not None:
        out["json_state"] = "json"
        v = _canon_verdict(obj.get("verdict"))
        ev = obj.get("evidence")
        if isinstance(ev, str):
            ev = [ev]
        ev = [e for e in (ev or []) if isinstance(e, str) and e.strip()]
        if v is not None and "evidence" in obj:
            out.update(verdict=v, evidence=ev, json_state="json",
                       schema_valid=True, parse_mode="strict_json")
            return out
        # well-formed JSON but wrong keys or bad verdict value
        out["json_state"] = "json_wrong_schema"
        if v is not None:
            out.update(verdict=v, evidence=ev, parse_mode="json_wrong_schema")
            return out
        for k in ("verdict", "label", "answer", "relation", "relationship", "result"):
            if k in obj:
                v2 = _canon_verdict(obj[k])
                if v2:
                    out.update(verdict=v2, evidence=ev, parse_mode="json_wrong_schema")
                    return out

    m = _VERDICT_RE.search(text)
    if m:
        out.update(verdict=_CANON[re.sub(r"[\s_-]", "", m.group(1)).lower()],
                   parse_mode="bare_label")
    return out


# ---- evidence matching (spec-exact) -------------------------------------
def covered_flags(gold_spans, pred_spans):
    """Per gold span: is its normalized text a contiguous substring of the
    concatenated normalized prediction?"""
    hay = norm(" ".join(pred_spans))
    return [bool(norm(g)) and norm(g) in hay for g in gold_spans]


def evidence_case(gold_spans, pred_spans):
    """-> ('TP'|'FN'|'FP'|'TN', fraction_of_gold_covered)."""
    g_non, p_non = len(gold_spans) > 0, len(pred_spans) > 0
    if g_non:
        flags = covered_flags(gold_spans, pred_spans)
        frac = sum(flags) / len(flags)
        return ("TP" if all(flags) else "FN"), frac
    return ("FP" if p_non else "TN"), 1.0


def jaccard(gold_spans, pred_spans):
    a = set(norm(" ".join(gold_spans)).split())
    b = set(norm(" ".join(pred_spans)).split())
    if not a and not b:
        return None
    return len(a & b) / len(a | b) if (a | b) else None
