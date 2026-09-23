/**
 * An uploaded playbook document -> the list of positions to check.
 *
 * Playbooks usually tag each position with a severity (HIGH / MEDIUM / LOW)
 * and follow it with rationale ("Why…", "Fallback…"). PDF extraction often
 * runs all of that onto one line, so a position is taken as the sentence
 * immediately BEFORE a severity marker, with the rationale cut off. Where a
 * document has no severity markers, sentences written as standing positions
 * ("we may…", "the agreement must…") are used instead.
 */

const SEVERITY = /\b(HIGH|MEDIUM|LOW|CRITICAL)\b/;
const SEVERITY_G = /\b(HIGH|MEDIUM|LOW|CRITICAL)\b/g;
// leading list markers: "1", "2.", "(a)", "-", "•", "A."
const LEAD = /^\s*(\(?[a-z0-9]{1,3}[.)]|[-•*•]|\d{1,3})\s+/i;
const SUPPORTING = /^(why|fallback|rationale|note|guidance|severity|example)\b[:\s]/i;
const MODAL = /\b(may|must|shall|should|will not|cannot|is not|are not)\b/i;
const HEADING = /^(section|appendix|schedule|part|annex|version|owner|effective|review cycle)\b/i;
// running heads, footers and contact lines
const BOILERPLATE = /(\b(internal|page \d+|v\d+\.\d+|©|all rights reserved)\b|@|https?:)/i;

const MIN_LEN = 25;
const MAX_LEN = 320;
export const MAX_POSITIONS = 40;

// PDF extraction often glues a heading onto the position that follows it
// ("SECTION E — INTELLECTUAL PROPERTY We may disclose…"); drop that prefix
const CAPS_PREFIX = /^(?:[A-Z][A-Z&/\-—·'’\s]{3,}?)(?=[A-Z][a-z])/;

const tidy = (s: string) =>
  s.replace(LEAD, "").replace(CAPS_PREFIX, "").trim().replace(/\s+/g, " ");

// a run of heading words glued mid-line, e.g. "…LOW note only SECTION A — SCOPE We may…"
const AFTER_CAPS = /(?:[A-Z][A-Z&/\-—·'’]{2,}[\s—·-]+)+([A-Z][a-z].*)$/;

/** Last complete sentence in a chunk of text, minus any heading glued before it. */
function lastSentence(chunk: string): string {
  const parts = chunk.split(/(?<=[.!?])\s+/).map(tidy).filter(Boolean);
  const last = parts.length ? parts[parts.length - 1] : "";
  if (last.length <= MAX_LEN) return last;
  return tidy(last.match(AFTER_CAPS)?.[1] ?? last);   // recover the tail
}

function usable(line: string): boolean {
  if (line.length < MIN_LEN || line.length > MAX_LEN) return false;
  if (SUPPORTING.test(line) || HEADING.test(line) || BOILERPLATE.test(line)) return false;
  const letters = line.replace(/[^A-Za-z]/g, "");
  return !(letters.length >= 4 && letters === letters.toUpperCase());   // not a heading
}

export function parsePlaybook(text: string): string[] {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const marked: string[] = [];

  for (const [i, line] of lines.entries()) {
    // severity on its own line: the position is the sentence just above it
    if (SEVERITY.test(line) && line.replace(SEVERITY, "").trim().length < 4) {
      const cand = lastSentence(lines[i - 1] ?? "");
      if (usable(cand)) marked.push(cand);
      continue;
    }
    // severity inline: one position per marker, each the sentence before it
    for (const m of line.matchAll(SEVERITY_G)) {
      const cand = lastSentence(line.slice(0, m.index));
      if (usable(cand)) marked.push(cand);
    }
  }

  const chosen = marked.length >= 3
    ? marked
    : lines.flatMap((l) => l.split(/(?<=[.!?])\s+/)).map(tidy)
        .filter((l) => usable(l) && MODAL.test(l));

  const seen = new Set<string>();
  const out: string[] = [];
  for (const p of chosen) {
    const key = p.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(/[.!?]$/.test(p) ? p : `${p}.`);
    if (out.length >= MAX_POSITIONS) break;
  }
  return out;
}
