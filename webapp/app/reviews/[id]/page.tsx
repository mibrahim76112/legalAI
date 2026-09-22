"use client";

import Link from "next/link";
import ReviewFrame from "@/components/ReviewFrame";
import { Badge, Tile } from "@/components/Badge";
import { counts, DECISION_LABEL, triageOrder } from "@/lib/review";
import type { Item, Review } from "@/lib/types";

export default function Overview() {
  return <ReviewFrame>{({ review }) => <Body r={review} />}</ReviewFrame>;
}

function Body({ r }: { r: Review }) {
  const c = counts(r);
  const order = triageOrder(r);
  const first = order.find((i) => !i.decision) ?? order[0];
  const top = r.compliance.filter((i) => i.tone === "red" || i.tone === "amb")
    .sort((a, b) => (a.tone === b.tone ? 0 : a.tone === "red" ? -1 : 1));
  const found = r.clauses.filter((i) => i.tone === "grn");
  const missing = r.clauses.filter((i) => i.tone === "gry");

  return (
    <div className="body-pad wide">
      <div className="next card">
        <div>
          <h2>{c.decided === c.total ? "Every finding has a decision" : `${c.total - c.decided} findings to review`}</h2>
          <p className="muted" style={{ margin: "4px 0 0" }}>
            {c.decided === c.total
              ? "Download the report to share it, or revisit any finding in Triage."
              : "Triage walks you through them one at a time, most serious first. Your decisions save as you go."}
          </p>
        </div>
        <span className="grow" />
        {c.decided === c.total
          ? <Link className="btn pri" href={`/reviews/${r.id}/report`}>Open report →</Link>
          : first && <Link className="btn pri" href={`/reviews/${r.id}/triage?item=${first.id}`}>
              {c.decided ? "Continue triage →" : "Start triage →"}</Link>}
      </div>

      <div className="tiles" style={{ margin: "18px 0 26px" }}>
        {r.tasks.includes("compliance") && <>
          <Tile tone="red" label="Needs attention" n={c.red} />
          <Tile tone="amb" label="Not addressed" n={c.amb} />
          <Tile tone="grn" label="Meets standard" n={c.grn} />
        </>}
        {r.tasks.includes("clauses") && <Tile tone="gry" label="Clauses found" n={c.found} />}
      </div>

      {r.tasks.includes("compliance") && (
        <section>
          <h2 className="sec">Top issues</h2>
          {top.length === 0
            ? <p className="muted">Nothing conflicts with or is missing from the playbook.</p>
            : <div className="list">{top.map((it) => <IssueRow key={it.id} r={r} it={it} />)}</div>}
        </section>
      )}

      {r.tasks.includes("clauses") && (
        <section style={{ marginTop: 26 }}>
          <h2 className="sec">Clause inventory</h2>
          <div className="list">{found.map((it) => <IssueRow key={it.id} r={r} it={it} />)}</div>
          {missing.length > 0 && (
            <p className="muted" style={{ marginTop: 10 }}>
              Not detected: {missing.map((m) => m.title).join(", ")}. The model may miss a clause, so &ldquo;not detected&rdquo; is not proof of absence.
            </p>
          )}
        </section>
      )}
    </div>
  );
}

function IssueRow({ r, it }: { r: Review; it: Item }) {
  return (
    <Link href={`/reviews/${r.id}/triage?item=${it.id}`} className="irow">
      <span className={`sd d-${it.tone}`} />
      <span className="tt">
        <span className="tl">{it.title}</span>
        {it.note && <span className="nt">{it.note}</span>}
      </span>
      <span className="meta">
        <Badge tone={it.tone}>{it.status}</Badge>
        {it.decision && <span className="dec">{DECISION_LABEL[it.decision]}</span>}
      </span>
    </Link>
  );
}
