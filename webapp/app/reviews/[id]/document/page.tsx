"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ReviewFrame, { type FrameProps } from "@/components/ReviewFrame";
import Doc, { type Mark } from "@/components/Doc";
import type { Review, Tone } from "@/lib/types";

export default function DocumentTab() {
  return <ReviewFrame full>{(p) => <Body {...p} />}</ReviewFrame>;
}

const LAYERS: { tone: Tone; label: string }[] = [
  { tone: "red", label: "Needs attention" },
  { tone: "amb", label: "Not addressed" },
  { tone: "grn", label: "Meets standard / found" },
];

function Body({ review }: FrameProps) {
  const router = useRouter();
  const [on, setOn] = useState<Record<string, boolean>>({ red: true, amb: true, grn: true });

  const byTone = useMemo(() => {
    const out: Record<string, Mark[]> = {};
    for (const it of [...review.compliance, ...review.clauses]) {
      out[it.tone] = (out[it.tone] ?? []).concat(it.evidence.map((s, i): Mark => ({
        key: `${it.id}:${i}`, start: s.start, end: s.end, tone: it.tone, itemId: it.id,
        label: `${it.status}: ${it.title}`,
      })));
    }
    return out;
  }, [review]);

  const marks = useMemo(
    () => LAYERS.flatMap((l) => (on[l.tone] ? byTone[l.tone] ?? [] : [])),
    [byTone, on]);

  // findings the model reports as absent have nothing to highlight, so they are
  // listed instead of being a layer that can never light up
  const absent = [...review.compliance, ...review.clauses].filter((i) => !i.evidence.length);

  return (
    <div className="docwrap">
      <div className="dh">
        <span style={{ fontWeight: 560, color: "var(--tx)" }}>Highlights</span>
        {LAYERS.filter((l) => (byTone[l.tone] ?? []).length).map((l) => (
          <button key={l.tone} className={`f${on[l.tone] ? " on" : ""}`}
                  onClick={() => setOn((o) => ({ ...o, [l.tone]: !o[l.tone] }))}>
            <span className={`d d-${l.tone}`} />{l.label}
            <span className="c">{(byTone[l.tone] ?? []).length}</span>
          </button>
        ))}
        <span className="grow" />
        <span>Click a highlight to open that finding</span>
      </div>
      {absent.length > 0 && <Absent review={review} items={absent} />}
      <Doc text={review.documentText} marks={marks}
           onMark={(m) => m.itemId && router.push(`/reviews/${review.id}/triage?item=${m.itemId}`)} />
    </div>
  );
}

function Absent({ review, items }: { review: Review; items: Review["compliance"] }) {
  const [open, setOpen] = useState(false);
  const positions = items.filter((i) => i.kind === "position");
  const clauses = items.filter((i) => i.kind === "clause");
  return (
    <div className="absent">
      <button className="hd" onClick={() => setOpen(!open)}>
        <span className="d d-amb" />
        <b>{items.length} not found in this document</b>
        <span className="muted">
          {positions.length > 0 && `${positions.length} playbook position${positions.length > 1 ? "s" : ""}`}
          {positions.length > 0 && clauses.length > 0 && " · "}
          {clauses.length > 0 && `${clauses.length} clause type${clauses.length > 1 ? "s" : ""}`}
          {" · nothing to highlight"}
        </span>
        <span className="chev">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <ul>
          {items.map((it) => (
            <li key={it.id}>
              <Link href={`/reviews/${review.id}/triage?item=${it.id}`}>{it.title}</Link>
              <span className="muted"> · {it.status}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
