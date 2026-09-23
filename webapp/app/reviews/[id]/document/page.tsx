"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import ReviewFrame, { type FrameProps } from "@/components/ReviewFrame";
import Doc, { type Mark } from "@/components/Doc";
import type { Tone } from "@/lib/types";

export default function DocumentTab() {
  return <ReviewFrame full>{(p) => <Body {...p} />}</ReviewFrame>;
}

const LAYERS: { tone: Tone; label: string }[] = [
  { tone: "red", label: "Needs attention" },
  { tone: "amb", label: "Not addressed" },
  { tone: "grn", label: "Meets standard / clause found" },
];

function Body({ review }: FrameProps) {
  const router = useRouter();
  const [on, setOn] = useState<Record<string, boolean>>({ red: true, amb: true, grn: true });
  const marks = useMemo(() => [...review.compliance, ...review.clauses]
    .filter((it) => on[it.tone])
    .flatMap((it) => it.evidence.map((s, i): Mark => ({
      key: `${it.id}:${i}`, start: s.start, end: s.end, tone: it.tone, itemId: it.id,
      label: `${it.status}: ${it.title}`,
    }))), [review, on]);

  return (
    <div className="docwrap">
      <div className="dh">
        <span style={{ fontWeight: 560, color: "var(--tx)" }}>Highlights</span>
        {LAYERS.map((l) => (
          <button key={l.tone} className={`f${on[l.tone] ? " on" : ""}`}
                  onClick={() => setOn((o) => ({ ...o, [l.tone]: !o[l.tone] }))}>
            <span className={`d d-${l.tone}`} />{l.label}
          </button>
        ))}
        <span className="grow" />
        <span>Click a highlight to open that finding</span>
      </div>
      <Doc text={review.documentText} marks={marks}
           onMark={(m) => m.itemId && router.push(`/reviews/${review.id}/triage?item=${m.itemId}`)} />
    </div>
  );
}
