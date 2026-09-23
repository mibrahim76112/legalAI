/**
 * Contract text -> blocks for a Markdown-style reading view.
 *
 * Offsets stay in the original plain text: findings carry character offsets
 * into documentText (from spans.py), so the reading view is a rendering of
 * that text, never a second copy with different positions.
 *
 * Heading rules mirror inference/extract.py `_unwrap`: a short line that starts
 * with a section number or is all caps.
 */

export interface Block {
  kind: "heading" | "para";
  start: number;
  end: number;
}

const SECTION = /^(\d+(\.\d+)*\.?|§\s*\d+|[A-Z]\.|ARTICLE\s+[IVXLC\d]+|Section\s+\d+)\s/;
const ENDS_SENTENCE = /[.;:!?]["'”’)]*$/;

function isHeading(line: string) {
  const t = line.trim();
  if (!t || t.length > 80) return false;
  const letters = t.replace(/[^A-Za-z]/g, "");
  if (letters.length >= 4 && letters === letters.toUpperCase()) return true;
  return SECTION.test(t) && t.length < 60 && !ENDS_SENTENCE.test(t);
}

export function blocks(text: string): Block[] {
  const out: Block[] = [];
  let pos = 0;
  for (const line of text.split("\n")) {
    const start = pos;
    const end = pos + line.length;
    pos = end + 1;
    const lead = line.length - line.trimStart().length;
    if (!line.trim()) continue;
    out.push({ kind: isHeading(line) ? "heading" : "para", start: start + lead, end });
  }
  return out;
}

/** The contract as a Markdown document: headings become "##", paragraphs stay. */
export function contractToMarkdown(title: string, text: string): string {
  const body = blocks(text).map((b) => {
    const s = text.slice(b.start, b.end).trim();
    return b.kind === "heading" ? `## ${s}` : s;
  });
  return [`# ${title}`, "", ...body.flatMap((s) => [s, ""])].join("\n");
}
