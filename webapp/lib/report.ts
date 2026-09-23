import { counts, DECISION_LABEL, triageOrder } from "./review";
import type { Item, Review } from "./types";

const quote = (s: string) => s.replace(/\s+/g, " ").trim();

function itemMd(it: Item): string[] {
  const out = [`### ${it.title}`, "", `**${it.status}**` +
    (it.decision ? ` · Reviewer: **${DECISION_LABEL[it.decision]}**` : " · Not yet reviewed"), ""];
  if (it.note) out.push(`**Why it matters:** ${it.note}`, "");
  if (it.comment) out.push(`**Reviewer comment:** ${it.comment}`, "");
  for (const e of it.evidence) out.push(`> ${quote(e.text)}`, "");
  return out;
}

/** The review as a Markdown report, grouped by severity. */
export function reviewToMarkdown(r: Review): string {
  const c = counts(r);
  const when = new Date(r.createdAt * 1000).toLocaleString();
  const items = triageOrder(r);
  const group = (title: string, pick: (i: Item) => boolean) => {
    const g = items.filter(pick);
    return g.length ? [`## ${title} (${g.length})`, "", ...g.flatMap(itemMd)] : [];
  };

  const lines = [
    `# Contract review: ${r.documentName}`, "",
    `- Reviewed: ${when}`,
    r.representing ? `- Acting for: ${r.representing}` : "",
    `- Checks: ${r.tasks.map((t) => (t === "compliance" ? "playbook positions" : "clause inventory")).join(", ")}`,
    `- Reviewer progress: ${c.decided} of ${c.total} findings decided${c.flagged ? `, ${c.flagged} flagged` : ""}`,
    "",
    "## Summary", "",
    "| | Count |", "|---|---|",
    ...(r.tasks.includes("compliance") ? [
      `| Needs attention | ${c.red} |`, `| Not addressed | ${c.amb} |`, `| Meets standard | ${c.grn} |`,
    ] : []),
    ...(r.tasks.includes("clauses") ? [`| Clauses found | ${c.found} |`, `| Clauses not detected | ${c.missing} |`] : []),
    "",
    ...group("Needs attention", (i) => i.kind === "position" && i.tone === "red"),
    ...group("Not addressed", (i) => i.kind === "position" && i.tone === "amb"),
    ...group("Meets standard", (i) => i.kind === "position" && i.tone === "grn"),
    ...group("Clauses found", (i) => i.kind === "clause" && i.tone === "grn"),
  ];
  const missing = items.filter((i) => i.kind === "clause" && i.tone === "gry");
  if (missing.length) {
    lines.push(`## Clauses not detected (${missing.length})`, "",
      ...missing.map((i) => `- ${i.title}${i.decision ? ` (${DECISION_LABEL[i.decision]})` : ""}`), "");
  }
  return lines.filter((l, i, a) => !(l === "" && a[i - 1] === "")).join("\n");
}

/** Save text as a file (local app, so a Blob link is fine). */
export function download(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const slug = (s: string) => s.replace(/[^\w-]+/g, "_").replace(/^_|_$/g, "") || "contract";
