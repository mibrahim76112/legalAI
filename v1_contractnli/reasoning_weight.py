#!/usr/bin/env python3
"""
Token-level loss reweighting for the reasoning field.

HYPOTHESIS UNDER TEST: the reasoning prose is irreducibly high-entropy -- its
loss cannot be driven down no matter the capacity (the LoRA rank sweep confirmed
r=64 does not close the 4x train-loss gap) -- yet it consumes ~80% of the
gradient. If so, the verdict and evidence signal is being diluted by tokens the
model cannot learn to predict, and down-weighting them should recover Arm A-like
performance while keeping the reasoning field.

Measured token shares of an Arm B target (Qwen3 tokenizer):
  reasoning ~74%, evidence ~19%, verdict ~2%, scaffolding ~5%

This is TOKEN-level, unlike the example-level WeightedTrainer already in
train_sft.py: that shifts how often a CLASS is seen, which cannot address an
imbalance INSIDE a single target.
"""
import json

SPLIT_KEY = '"verdict"'


def split_target(target_text):
    """-> (reasoning_part, answer_part). Split at the verdict key, so the
    reasoning prose and the {verdict, evidence} payload are separable."""
    i = target_text.find(SPLIT_KEY)
    if i < 0:
        return "", target_text
    return target_text[:i], target_text[i:]


def build_weighted_target(tok, target_text, reasoning_weight):
    """-> (token_ids, per_token_weights). Tokenizes the two halves separately so
    the boundary is exact; the sequence is then built FROM those pieces, so no
    retokenization mismatch can occur."""
    pre, post = split_target(target_text)
    a = tok(pre, add_special_tokens=False)["input_ids"] if pre else []
    b = tok(post, add_special_tokens=False)["input_ids"]
    ids = a + b
    w = [reasoning_weight] * len(a) + [1.0] * len(b)
    return ids, w


def describe(tok, target_text):
    pre, post = split_target(target_text)
    a = len(tok(pre, add_special_tokens=False)["input_ids"]) if pre else 0
    b = len(tok(post, add_special_tokens=False)["input_ids"])
    return {"reasoning_tokens": a, "answer_tokens": b,
            "reasoning_share": a / (a + b) if a + b else 0.0}
