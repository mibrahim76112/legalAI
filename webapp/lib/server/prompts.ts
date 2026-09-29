/**
 * The prompts the adapter was trained on, copied verbatim from
 * macbundle/lib/prompts.py. Changing a character puts the model
 * off-distribution, so treat these as fixed.
 *
 * Task 3 uses the SHORT system prompt that the validation rows use, not
 * prompts.TASK3_SYSTEM: on validation_60 it agreed with the cluster far
 * better (token Jaccard 0.60 vs 0.39).
 */

export const TASK1_SYSTEM =
  "You are a junior legal assistant reviewing a contract. Given the CONTRACT " +
  "and the QUESTION, determine whether the contract supports the stated position " +
  "(Entailment), conflicts with it (Contradiction), or does not address it " +
  "(NotMentioned). Return the exact sentence(s) from the contract that justify " +
  "your answer, or an empty evidence list if none apply. Respond with JSON only, " +
  'in the form {"verdict": "...", "evidence": [...]}.';

export const task1User = (text: string, question: string) =>
  `CONTRACT: ${text}\n\nQUESTION: ${question}`;

export const TASK2_SYSTEM =
  "You are a contract review assistant. Given the CONTRACT and a CLAUSE " +
  "CATEGORY, determine whether the contract contains a clause of that " +
  "category. If it does, return the exact sentence(s) from the contract that " +
  "constitute it; if it does not, return an empty evidence list. Respond with " +
  'JSON only, in the form {"present": true|false, "evidence": [...]}.';

export const task2User = (text: string, category: string) =>
  `CONTRACT: ${text}\n\nCLAUSE CATEGORY: ${category}`;

export const TASK3_SYSTEM =
  "You are a contract review assistant. Given a clause and its category, write " +
  "EXACTLY ONE sentence for a business reader explaining what the clause does in " +
  "practice and what brings it into effect.";

export const task3User = (cat: string, clause: string) =>
  `CLAUSE CATEGORY: ${cat}\n\nCLAUSE: ${clause}`;

export const PLAYBOOK_POSITIONS = [
  "Receiving Party may acquire information similar to Confidential Information from a third party.",
  "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
  "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
  "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
  "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
  "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
  "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
  "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
  "Receiving Party shall not solicit some of Disclosing Party's representatives.",
  "Confidential Information shall only include technical information.",
  "Receiving Party may independently develop information similar to Confidential Information.",
  "Receiving Party may create a copy of some Confidential Information in some circumstances.",
  "Confidential Information may include verbally conveyed information.",
  "All Confidential Information shall be expressly identified by the Disclosing Party.",
  "Agreement shall not grant Receiving Party any right to Confidential Information.",
  "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
  "Some obligations of Agreement may survive termination of Agreement.",
];

export const CLAUSE_CATEGORIES = [
  "Anti-Assignment", "Audit Rights", "Cap On Liability", "Change Of Control",
  "Covenant Not To Sue", "Exclusivity", "Insurance", "Ip Ownership Assignment",
  "License Grant", "Minimum Commitment", "Non-Compete", "Non-Transferable License",
  "Notice Period To Terminate Renewal", "Post-Termination Services", "Renewal Term",
  "Revenue/Profit Sharing", "Termination For Convenience", "Uncapped Liability",
];

// A not-addressed position has no clause text for the model to explain.
export const SILENT_NOTE =
  "The contract is silent on this, so nothing in it protects you here.";

export const POSITION_STATUS = {
  Entailment: ["Meets standard", "grn", "The contract supports this position."],
  Contradiction: ["Needs attention", "red", "The contract appears to conflict with this position."],
  NotMentioned: ["Not addressed", "amb", "The contract does not address this."],
} as const;
