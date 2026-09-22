#!/usr/bin/env python3
"""
THE normalizer. Single definition, imported everywhere.

Previously duplicated: doc_harness.norm (null-safe) and build_doc_sft.norm (not).
build_doc_sft runs at DATASET CONSTRUCTION and doc_harness at EVAL, so any
divergence between them makes training-time and eval-time span matching disagree
silently -- a bug class that is very hard to find after the fact. Consolidated to
the null-safe form; the only behaviour change is that None no longer raises.

Spec (unchanged): lowercase, collapse all whitespace runs to a single space,
strip. Nothing else.
"""


def norm(s):
    return " ".join((s or "").lower().split())
