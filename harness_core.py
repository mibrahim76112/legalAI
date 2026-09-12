#!/usr/bin/env python3
"""Prompt construction + answer parsing. Identical for all four models."""
import json, re

CLASSES = ["Entailment", "Contradiction", "NotMentioned"]

# Two system prompts are in play and they are NOT the same string:
#
#  "spec" - given verbatim in the screen brief. Names the exact JSON shape.
#  "sft"  - what 02_build_sft_data.py bakes into the training targets.
#
# The screen ranks models, so all four must share one prompt; "spec" is the
# brief's choice and is the default. But these base numbers are also meant to
# be the "base" column of the final fine-tuned-vs-base table, and that column
# is only a clean control if it used the prompt the fine-tune was trained on.
# Hence the flag: re-run with --system sft to get a prompt-matched baseline.
SYSTEM_PROMPTS = {
    "spec": (
        "You are reviewing a clause from a Non-Disclosure Agreement against a company "
        "policy position. Given the CLAUSE and the POLICY, determine whether the clause "
        "supports the policy (Entailment), conflicts with it (Contradiction), or does not "
        "address it (NotMentioned). Respond with JSON only, in the form "
        '{"verdict": "<Entailment|Contradiction|NotMentioned>"}.'
    ),
    "sft": (
        "You are reviewing a clause from a Non-Disclosure Agreement against a "
        "company policy position. Given the CLAUSE and the POLICY, determine "
        "whether the clause supports the policy (Entailment), conflicts with it "
        "(Contradiction), or does not address it (NotMentioned). Answer with a "
        "single JSON object and nothing else."
    ),
}
SYSTEM = SYSTEM_PROMPTS["spec"]


def user_turn(premise: str, hypothesis: str) -> str:
    return f"CLAUSE: {premise}\nPOLICY: {hypothesis}"


def build_messages(premise, hypothesis, fewshot=None, system=SYSTEM):
    """fewshot=None -> zero-shot (condition A); list -> 3-shot (condition B)."""
    msgs = [{"role": "system", "content": system}]
    for ex in (fewshot or []):
        msgs.append({"role": "user", "content": user_turn(ex["premise"], ex["hypothesis"])})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"verdict": ex["verdict"]})})
    msgs.append({"role": "user", "content": user_turn(premise, hypothesis)})
    return msgs


# ---- parsing -------------------------------------------------------------
# Canonical spellings, plus the variants models actually emit. Matching is
# case-insensitive; "not mentioned"/"not_mentioned" fold onto NotMentioned.
_CANON = {
    "entailment": "Entailment", "entailed": "Entailment", "entails": "Entailment",
    "contradiction": "Contradiction", "contradicts": "Contradiction",
    "contradictory": "Contradiction",
    "notmentioned": "NotMentioned", "not mentioned": "NotMentioned",
    "not_mentioned": "NotMentioned", "not-mentioned": "NotMentioned",
    "neutral": "NotMentioned",
}
_JSON_RE = re.compile(r'\{[^{}]*"verdict"\s*:\s*"?([A-Za-z _-]+)"?[^{}]*\}', re.I)
_BARE_RE = re.compile(r'\b(entailment|contradiction|not[ _-]?mentioned|neutral)\b', re.I)
# Reasoning models wrap their answer; strip the thinking block before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _canon(s):
    return _CANON.get(s.strip().lower().replace('"', ""))


def parse_verdict(raw: str):
    """-> (verdict|None, json_valid: bool, how: str).

    json_valid means the model actually emitted the requested JSON object,
    which we report separately from whether the verdict was right.
    """
    if raw is None:
        return None, False, "empty"
    text = _THINK_RE.sub(" ", raw).strip()
    if not text:
        return None, False, "empty"

    # 1. strict: the whole reply is the requested object
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "verdict" in obj:
            v = _canon(str(obj["verdict"]))
            if v:
                return v, True, "strict_json"
    except Exception:
        pass

    # 2. a JSON object embedded in prose / fences
    m = _JSON_RE.search(text)
    if m:
        v = _canon(m.group(1))
        if v:
            return v, True, "embedded_json"

    # 3. bare label anywhere - counts as parsed but NOT json_valid
    m = _BARE_RE.search(text)
    if m:
        v = _canon(m.group(1))
        if v:
            return v, False, "bare_label"

    return None, False, "unparseable"
