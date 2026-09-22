"use client";

import { useEffect, useRef } from "react";
import type { Span, Tone } from "@/lib/types";

export default function Doc({ text, spans, tone, cur, title }:
  { text: string; spans: Span[]; tone: Tone; cur: number; title: string | null }) {
  const ref = useRef<HTMLElement | null>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [cur, spans]);

  const ord = spans.map((s, i) => ({ ...s, i })).sort((a, b) => a.start - b.start);
  const keep: typeof ord = [];
  let last = -1;
  for (const s of ord) if (s.start >= last) { keep.push(s); last = s.end; }

  const out: React.ReactNode[] = [];
  let p = 0;
  keep.forEach((s) => {
    if (s.start > p) out.push(text.slice(p, s.start));
    const on = s.i === cur;
    out.push(
      <mark key={s.start} className={`m-${tone}${on ? " cur" : ""}`}
            ref={on ? (e) => { ref.current = e; } : undefined}>
        {text.slice(s.start, s.end)}
      </mark>
    );
    p = s.end;
  });
  if (p < text.length) out.push(text.slice(p));

  return (
    <>
      <div className="dh">
        <span style={{ fontWeight: 560, color: "var(--tx)" }}>Document</span>
        {title ? (
          <span>· {spans.length} passage{spans.length > 1 ? "s" : ""} highlighted</span>
        ) : (
          <span>· select a finding to highlight it</span>
        )}
      </div>
      <div className="doc">{out}</div>
    </>
  );
}
