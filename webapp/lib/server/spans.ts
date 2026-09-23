/**
 * Quoted model output -> character offsets in the original contract.
 *
 * Port of macbundle/lib/spans.py, which round-tripped 745/745 gold spans on
 * the cluster. The invariant from that file holds here too: normWithMap(s)[0]
 * must equal norm(s) exactly, because the per-character index back into the
 * original is the only reason offsets are recoverable. `sameNorm` asserts it.
 */

export interface Span { text: string; start: number; end: number }

/** Lowercase, collapse whitespace runs to one space, trim. Nothing else. */
export function norm(s: string): string {
  return (s || "").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

/** norm(s) plus, for each character of the result, its index in s. */
export function normWithMap(s: string): [string, number[]] {
  const out: string[] = [];
  const idx: number[] = [];
  let prevSpace = true;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (/\s/.test(ch)) {
      if (!prevSpace && out.length) { out.push(" "); idx.push(i); }
      prevSpace = true;
    } else {
      out.push(ch.toLowerCase());
      idx.push(i);
      prevSpace = false;
    }
  }
  while (out.length && out[out.length - 1] === " ") { out.pop(); idx.pop(); }
  return [out.join(""), idx];
}

export const sameNorm = (s: string) => normWithMap(s)[0] === norm(s);

const EDGE = /^[.,;:!?'"’—\- ]+|[.,;:!?'"’—\- ]+$/g;

/** Offsets of `span` inside the text whose normalized form is hayNorm/hayIdx. */
export function locate(span: string, hayNorm: string, hayIdx: number[]): [number, number] | null {
  // mirrors _grounded's tolerance for edge punctuation a model adds when it
  // quotes a heading
  for (const cand of [norm(span), norm(span).replace(EDGE, "")]) {
    if (!cand) continue;
    const p = hayNorm.indexOf(cand);
    if (p >= 0) return [hayIdx[p], hayIdx[p + cand.length - 1] + 1];
  }
  return null;
}

/** Locatable quotes as spans; unlocatable ones are dropped and counted. */
export function spansOf(doc: string, texts: string[]): { spans: Span[]; dropped: number } {
  const [hayNorm, hayIdx] = normWithMap(doc);
  const seen = new Set<string>();
  const spans: Span[] = [];
  let dropped = 0;
  for (const t of texts) {
    const k = norm(t);
    if (!k || seen.has(k)) continue;
    seen.add(k);
    const loc = locate(t, hayNorm, hayIdx);
    if (loc) spans.push({ text: t, start: loc[0], end: loc[1] });
    else dropped++;
  }
  return { spans, dropped };
}
