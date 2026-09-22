import type { Decision, Item, Review } from "./types";

/** Findings in triage order: needs attention, not addressed, meets standard, clauses found, not detected. */
export function triageOrder(r: Review): Item[] {
  const rank = { red: 0, amb: 1, grn: 2, gry: 3 };
  const pos = [...r.compliance].sort((a, b) => rank[a.tone] - rank[b.tone]);
  const cls = [...r.clauses].sort((a, b) => rank[a.tone] - rank[b.tone]);
  return [...pos, ...cls];
}

export function counts(r: Review) {
  const all = [...r.compliance, ...r.clauses];
  return {
    red: r.compliance.filter((i) => i.tone === "red").length,
    amb: r.compliance.filter((i) => i.tone === "amb").length,
    grn: r.compliance.filter((i) => i.tone === "grn").length,
    found: r.clauses.filter((i) => i.tone === "grn").length,
    missing: r.clauses.filter((i) => i.tone === "gry").length,
    decided: all.filter((i) => i.decision).length,
    flagged: all.filter((i) => i.decision === "flag").length,
    total: all.length,
  };
}

export const DECISION_LABEL: Record<Exclude<Decision, null>, string> = {
  ok: "Accepted", flag: "Flagged", skip: "Dismissed",
};
