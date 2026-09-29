#!/usr/bin/env python3
"""Span location: map a model's quoted text back to character offsets.

Extracted from v2/cuad_windowed.py. Validated on the cluster: 745/745 gold
spans round-tripped, plus 227/227 sampled from window-overlap regions, where
round-trip means slicing the ORIGINAL contract at the computed offsets and
matching the span after normalization.

norm_with_map must agree with text_norm.norm exactly; assert it if you touch it.
"""
from text_norm import norm
from reasoning_filters import _grounded


def norm_with_map(s):
    """norm(s) plus a per-character map back into s.

    Must agree with text_norm.norm exactly; asserted by the caller. Needed
    because _grounded works on normalized text but coordinates must be
    reported in the ORIGINAL string.
    """
    out, idx, prev_space = [], [], True
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def locate(span, hay_raw, hay_norm, hay_idx):
    """-> (start, end) char offsets of `span` inside hay_raw, or None.

    Mirrors _grounded's tolerance for the edge punctuation a model adds when
    quoting a heading, so anything _grounded accepts can also be located.
    """
    for cand in (norm(span), norm(span).strip(".,;:!?’'\"-— ")):
        if not cand:
            continue
        p = hay_norm.find(cand)
        if p >= 0:
            return hay_idx[p], hay_idx[p + len(cand) - 1] + 1
    return None



def spans_of(doc, texts):
    """-> [{text,start,end}] for each locatable span; silently drops the rest."""
    dn, di = norm_with_map(doc)
    out = []
    for t in texts:
        loc = locate(t, doc, dn, di)
        if loc:
            out.append({"text": t, "start": loc[0], "end": loc[1]})
    return out
