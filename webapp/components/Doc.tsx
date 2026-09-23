"use client";

import { useEffect, useMemo, useRef } from "react";
import { blocks } from "@/lib/structure";
import type { Tone } from "@/lib/types";

export interface Mark {
  key: string;        // unique per highlighted passage
  start: number;
  end: number;
  tone: Tone;
  itemId?: string;
  label?: string;     // tooltip
}

/**
 * The contract as a reading document: headings and paragraphs from
 * lib/structure, with passages highlighted by their character offsets into
 * the original text. `current` scrolls its mark into view.
 */
export default function Doc({ text, marks, current, onMark }: {
  text: string; marks: Mark[]; current?: string | null; onMark?: (m: Mark) => void;
}) {
  const ref = useRef<HTMLElement | null>(null);
  const bs = useMemo(() => blocks(text), [text]);

  // overlapping marks can't nest in the DOM; keep the earliest of each overlap
  const kept = useMemo(() => {
    const out: Mark[] = [];
    let last = -1;
    for (const m of [...marks].sort((a, b) => a.start - b.start || b.end - a.end)) {
      if (m.start >= last) { out.push(m); last = m.end; }
    }
    return out;
  }, [marks]);

  useEffect(() => {
    // on first paint the scroll container has no layout yet, so a scroll here
    // is a no-op; wait one frame before scrolling the passage into view
    const id = requestAnimationFrame(() =>
      ref.current?.scrollIntoView({ behavior: "smooth", block: "center" }));
    return () => cancelAnimationFrame(id);
  }, [current]);

  let firstOfCurrent = true;
  const render = (b: { start: number; end: number }) => {
    const out: React.ReactNode[] = [];
    let p = b.start;
    for (const m of kept) {
      if (m.end <= b.start || m.start >= b.end) continue;
      const s = Math.max(m.start, b.start), e = Math.min(m.end, b.end);
      if (s > p) out.push(text.slice(p, s));
      const on = m.key === current;
      const attach = on && firstOfCurrent;
      if (attach) firstOfCurrent = false;
      out.push(
        <mark key={`${m.key}@${s}`} className={`m-${m.tone}${on ? " cur" : ""}${onMark ? " click" : ""}`}
              title={m.label} onClick={onMark ? () => onMark(m) : undefined}
              ref={attach ? (el) => { ref.current = el; } : undefined}>
          {text.slice(s, e)}
        </mark>
      );
      p = e;
    }
    if (p < b.end) out.push(text.slice(p, b.end));
    return out;
  };

  return (
    <article className="doc">
      {bs.map((b) => b.kind === "heading"
        ? <h3 key={b.start}>{render(b)}</h3>
        : <p key={b.start}>{render(b)}</p>)}
    </article>
  );
}
