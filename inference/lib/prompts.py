#!/usr/bin/env python3
"""The exact prompts the adapter was trained on, for all three tasks.

Copied from their sources of truth, not retyped:
  task 1  v1_contractnli/doc_harness.py
  task 2  v2/build_cuad.py
  task 3  v2/gen_risk_notes.py

Deviating from these at inference puts the model off-distribution: it was
trained on these strings, including the JSON shape in the system message.

Chat template: apply_chat_template(msgs, add_generation_prompt=True) and, for
models whose template supports it, enable_thinking=False. Detect support by
rendering the template BOTH ways and comparing the text -- a template that
silently ignores an unknown kwarg will otherwise look supported.
Greedy decoding, max_new_tokens 1024.
"""

# ---------------- Task 1: compliance verdict (ContractNLI) ------------
TASK1_SYSTEM = (
    "You are a junior legal assistant reviewing a contract. Given the CONTRACT "
    "and the QUESTION, determine whether the contract supports the stated position "
    "(Entailment), conflicts with it (Contradiction), or does not address it "
    "(NotMentioned). Return the exact sentence(s) from the contract that justify "
    "your answer, or an empty evidence list if none apply. Respond with JSON only, "
    "in the form {\"verdict\": \"...\", \"evidence\": [...]}."
)
TASK1_USER = "CONTRACT: {text}\n\nQUESTION: {question}"

# ---------------- Task 2: clause identification (CUAD) ----------------
TASK2_SYSTEM = (
    "You are a contract review assistant. Given the CONTRACT and a CLAUSE "
    "CATEGORY, determine whether the contract contains a clause of that "
    "category. If it does, return the exact sentence(s) from the contract that "
    "constitute it; if it does not, return an empty evidence list. Respond with "
    'JSON only, in the form {"present": true|false, "evidence": [...]}.'
)
TASK2_USER = "CONTRACT: {text}\n\nCLAUSE CATEGORY: {category}"

# ---------------- Task 3: risk note (synthetic) -----------------------
TASK3_SYSTEM = (
    "You are a contract review assistant. A business reader with no legal "
    "training is about to sign a contract and has been shown one clause from "
    "it.\n\n"
    "In EXACTLY ONE sentence, tell that reader two things: what the clause "
    "does to them in practice, and what event or condition brings it into "
    "effect.\n\n"
    "Vary your phrasing between notes; do not fall into a fixed formula. Do "
    "NOT begin with the words 'This clause'. Do not recommend legal action. "
    "Do not suggest negotiating, amending, or consulting counsel. Do not state "
    "any fact that is not supported by the clause text itself. Write one "
    "sentence and nothing else -- no preamble, no bullet points, no quotation "
    "marks around the whole sentence."
)
TASK3_USER = "CLAUSE CATEGORY: {cat}\n\nCLAUSE: {clause}"

PLAYBOOK_POSITIONS = [
    'Receiving Party may acquire information similar to Confidential Information from a third party.',
    'Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).',
    'Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.',
    'Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.',
    "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    'Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.',
    'Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.',
    "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
    "Receiving Party shall not solicit some of Disclosing Party's representatives.",
    'Confidential Information shall only include technical information.',
    'Receiving Party may independently develop information similar to Confidential Information.',
    'Receiving Party may create a copy of some Confidential Information in some circumstances.',
    'Confidential Information may include verbally conveyed information.',
    'All Confidential Information shall be expressly identified by the Disclosing Party.',
    'Agreement shall not grant Receiving Party any right to Confidential Information.',
    'Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.',
    'Some obligations of Agreement may survive termination of Agreement.',
]

CLAUSE_CATEGORIES = [
    'Anti-Assignment',
    'Audit Rights',
    'Cap On Liability',
    'Change Of Control',
    'Covenant Not To Sue',
    'Exclusivity',
    'Insurance',
    'Ip Ownership Assignment',
    'License Grant',
    'Minimum Commitment',
    'Non-Compete',
    'Non-Transferable License',
    'Notice Period To Terminate Renewal',
    'Post-Termination Services',
    'Renewal Term',
    'Revenue/Profit Sharing',
    'Termination For Convenience',
    'Uncapped Liability',
]
