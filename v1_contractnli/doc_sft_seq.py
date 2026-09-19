#!/usr/bin/env python3
"""
Canonical training-sequence construction for Phase 2. Phase 2 imports THIS -
the code that gets verified is the code that trains.

Two silent failure modes this exists to prevent:

1. TRAIN/INFER TEMPLATE MISMATCH. Building with
   apply_chat_template(full_messages) does not necessarily reproduce the
   inference prompt. gemma-4-12B-it is a live example: its training render goes
   straight to the answer, while add_generation_prompt=True inserts an empty
   <|channel>thought<channel|>. Fine-tuning on one and serving the other means
   the model never saw the tokens immediately preceding its answer.
   Fix: build the sequence AS prompt + target, where `prompt` is literally the
   inference prompt. Alignment becomes structural, not coincidental.

2. UNMASKED PROMPT LOSS. Targets are ~50-100 tokens against ~2,500 tokens of
   contract. Training without masking spends ~96% of the loss on reproducing
   contract text and ~4% on the task. It does not crash; it just quietly
   wastes the run. Fix: label -100 for every prompt token.
"""
IGNORE_INDEX = -100


def ids(enc):
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return list(enc)


def detect_think_kwarg(tok, msgs):
    """Probe BOTH directions; a template whose default is already non-thinking
    renders identically for undefined and False (Gemma 4)."""
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for name in ("enable_thinking", "thinking"):
        try:
            on = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, **{name: True})
            off = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, **{name: False})
        except Exception:
            continue
        if on != off:
            return name, (base == on)
    return None, None


def build_example(tok, messages, think_kwarg=None, max_seq_len=8192,
                  append_eos=True):
    """-> dict(input_ids, labels, n_prompt, n_target, truncated).

    Non-thinking is forced: Phase 2 trains and Phase 3 infers in that mode.
    """
    kw = {think_kwarg: False} if think_kwarg else {}
    prompt_msgs = [m for m in messages if m["role"] != "assistant"]
    target_text = [m for m in messages if m["role"] == "assistant"][0]["content"]

    prompt_ids = ids(tok.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, **kw))
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
    if append_eos and tok.eos_token_id is not None:
        target_ids = target_ids + [tok.eos_token_id]

    input_ids = prompt_ids + target_ids
    # -100 over the whole prompt so loss is computed on the target only.
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)

    truncated = False
    if len(input_ids) > max_seq_len:
        # Truncate the PROMPT from the left, never the target: dropping target
        # tokens would train the model to emit malformed JSON. Contract head
        # is the least load-bearing region for this task.
        overflow = len(input_ids) - max_seq_len
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + target_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
            truncated = True
        else:
            return None      # target alone exceeds the budget: drop the row

    assert len(input_ids) == len(labels)
    assert all(l == IGNORE_INDEX for l in labels[:len(prompt_ids)])
    assert labels[len(prompt_ids):] == list(target_ids)
    return {"input_ids": input_ids, "labels": labels,
            "n_prompt": len(prompt_ids), "n_target": len(target_ids),
            "truncated": truncated}
