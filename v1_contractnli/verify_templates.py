#!/usr/bin/env python3
"""
PHASE 1 pre-flight: render each model's chat template verbatim and report
exactly where generation begins.

Why this runs before any GPU work: malformed output is far more often a
template bug than a bad model. Two specific traps we know about:

  1. transformers>=5 returns a BatchEncoding from tokenize=True, not a list.
  2. Templates SILENTLY IGNORE unknown kwargs, so probing enable_thinking with
     a try/except TypeError reports false support. Detect by comparing the
     rendered text instead.

Qwen3 pre-fills an EMPTY <think></think> block when thinking is disabled. That
is part of the PROMPT, not the generation, so the model's output starts after
it and the parser must not try to strip a block that was never generated.
"""
import json, argparse

CONTRACT = "SECTION 1. The Recipient shall keep all Confidential Information secret."
QUESTION = "Receiving Party shall not disclose Confidential Information to third parties."
SYSTEM = "You are a junior legal assistant reviewing a contract. Respond with JSON only."


def detect_think_kwarg(tok, msgs):
    """Return (kwarg_name, default_is_thinking) or (None, None).

    Probing only `False` is NOT sufficient. Jinja treats an undefined variable
    as falsy, so for a template whose default is already non-thinking (Gemma 4:
    `{%- if not enable_thinking -%}`), render(undefined) == render(False) and
    the kwarg looks unsupported. Probe BOTH directions and compare them to each
    other, then work out which one the default matches.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file", default="models_doc.txt")
    ap.add_argument("--tail", type=int, default=130)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CONTRACT: {CONTRACT}\n\nQUESTION: {QUESTION}"}]

    findings = {}
    for m in [l.strip() for l in open(args.models_file) if l.strip()]:
        print("=" * 78)
        print(m)
        print("=" * 78)
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            print(f"  UNAVAILABLE: {type(e).__name__} - {str(e)[:90]}\n")
            findings[m] = {"available": False}
            continue

        kw, default_thinking = detect_think_kwarg(tok, msgs)
        render = {}
        for mode, extra in (("thinking_OFF", {kw: False} if kw else {}),
                            ("thinking_ON", {kw: True} if kw else {})):
            render[mode] = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **extra)

        off = render["thinking_OFF"]
        has_sys = SYSTEM[:30] in off
        prefilled = "<think>" in off
        empty_prefill = "<think>" in off and "</think>" in off

        print(f"  thinking kwarg   : {kw or 'NONE (no toggle in this template)'}")
        if kw:
            print(f"  template DEFAULT : {'THINKING' if default_thinking else 'non-thinking'}"
                  f"  (so condition A must pass {kw}=False explicitly"
                  f"{'' if default_thinking else '; default already correct'})")
        print(f"  system turn kept : {has_sys}")
        print(f"  <think> prefilled in PROMPT (thinking OFF): {prefilled}"
              + ("  [EMPTY block]" if empty_prefill else ""))
        print(f"  rendered length  : {len(off)} chars")
        print(f"\n  --- tail of prompt, thinking OFF (generation starts after this) ---")
        print("  " + repr(off[-args.tail:]))
        if kw:
            print(f"  --- tail, thinking ON ---")
            print("  " + repr(render['thinking_ON'][-args.tail:]))
        print()

        findings[m] = {"available": True, "think_kwarg": kw,
                       "default_is_thinking": default_thinking,
                       "system_supported": has_sys,
                       "think_prefilled_when_off": prefilled,
                       "empty_think_block": empty_prefill,
                       "prompt_tail_off": off[-args.tail:],
                       "prompt_tail_on": render["thinking_ON"][-args.tail:] if kw else None}

    json.dump(findings, open("data/template_findings.json", "w"), indent=2)
    print("-> data/template_findings.json")


if __name__ == "__main__":
    main()
