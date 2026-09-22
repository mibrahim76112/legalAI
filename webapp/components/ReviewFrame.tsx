"use client";

import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import Shell from "./Shell";
import { Badge } from "./Badge";
import { api, useReview } from "@/lib/api";
import { counts } from "@/lib/review";
import type { Decision, Item, Review } from "@/lib/types";

export interface FrameProps {
  review: Review;
  decide: (item: Item, d: Decision, comment: string | null) => Promise<void>;
}

const TABS = [
  { seg: "", label: "Overview" },
  { seg: "/triage", label: "Triage" },
  { seg: "/document", label: "Document" },
  { seg: "/report", label: "Report" },
];

/**
 * Header, tab bar and loading/running/error states shared by every review
 * page. Children render only once the review is done.
 */
export default function ReviewFrame({ children, full }: {
  children: (p: FrameProps) => React.ReactNode; full?: boolean;
}) {
  const { id } = useParams<{ id: string }>();
  const path = usePathname();
  const { review, error, reload, decide } = useReview(id);
  const c = review && review.status === "done" ? counts(review) : null;

  return (
    <Shell>
      <div className="phead rhead">
        <div style={{ minWidth: 0 }}>
          <div className="crumb"><Link href="/reviews">Reviews</Link> / {review?.documentName ?? "…"}</div>
          <h1 className="ell">{review?.documentName ?? "Loading…"}</h1>
          <div className="sub">
            {review?.representing && <>Acting for {review.representing} · </>}
            {review?.sample ? "Sample from the cluster evaluation" : review && new Date(review.createdAt * 1000).toLocaleString()}
          </div>
        </div>
        <span className="grow" />
        {c && (
          <div className="rprog">
            <div className="muted">{c.decided} of {c.total} reviewed</div>
            <div className="track"><i style={{ width: `${(c.decided / Math.max(c.total, 1)) * 100}%` }} /></div>
          </div>
        )}
      </div>
      {review?.status === "done" && (
        <nav className="tabs">
          {TABS.map((t) => {
            const href = `/reviews/${id}${t.seg}`;
            const on = t.seg ? path.startsWith(href) : path === href;
            return <Link key={t.seg} href={href} className={on ? "on" : ""}>{t.label}</Link>;
          })}
        </nav>
      )}
      <div className={full ? "rbody full" : "rbody"}>
        {!review && error && <Notice title="Can't load this review" body={error} />}
        {!review && !error && <div className="empty">Loading…</div>}
        {review && (review.status === "queued" || review.status === "running") && <Running review={review} />}
        {review && (review.status === "error" || review.status === "interrupted") &&
          <Failed review={review} onRerun={reload} />}
        {review?.status === "done" && children({ review, decide })}
      </div>
    </Shell>
  );
}

function Notice({ title, body, children }: { title: string; body: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="body-pad narrow">
      <div className="card" style={{ padding: "18px 22px" }}>
        <h2>{title}</h2>
        <p className="p" style={{ marginTop: 8, color: "var(--tx2)" }}>{body}</p>
        {children}
      </div>
    </div>
  );
}

function Running({ review }: { review: Review }) {
  const p = review.progress;
  const stages = [
    { k: "queued", l: "Reading the contract", on: true },
    { k: "positions", l: "Checking playbook positions", on: review.tasks.includes("compliance") },
    { k: "clauses", l: "Finding clauses", on: review.tasks.includes("clauses") },
    { k: "notes", l: "Explaining what each finding means", on: true },
  ].filter((s) => s.on);
  const cur = Math.max(0, stages.findIndex((s) => s.k === (p?.stage ?? "queued")));
  return (
    <div className="body-pad narrow">
      <div className="card" style={{ padding: "18px 22px" }}>
        <h2>Analyzing</h2>
        <p className="muted" style={{ margin: "4px 0 10px" }}>
          {review.status === "queued" ? "Waiting for the model (it may still be loading, or another review is running)."
            : "You can leave this page; the review keeps running and is saved when it finishes."}
        </p>
        {stages.map((s, k) => (
          <div key={s.k} className={`stg ${k < cur ? "done" : k === cur && review.status === "running" ? "now" : ""}`}>
            <span className="ic">{k < cur ? "✓" : ""}</span>{s.l}
            {k === cur && p && p.total > 0 && (
              <span className="muted" style={{ marginLeft: 8 }}>
                {p.done} of {p.total}
                {p.windows && p.windows > 1 && s.k !== "notes" && ` · part ${p.window} of ${p.windows}`}
              </span>
            )}
          </div>
        ))}
        <div className="track"><i style={{ width: `${p && p.total ? (p.done / p.total) * 100 : 0}%` }} /></div>
      </div>
    </div>
  );
}

function Failed({ review, onRerun }: { review: Review; onRerun: () => void }) {
  const router = useRouter();
  const [err, setErr] = useState<string | null>(null);
  const interrupted = review.status === "interrupted";
  return (
    <Notice title={interrupted ? "This review was interrupted" : "This review failed"}
            body={interrupted ? "The model server stopped before it finished. Nothing was lost except progress; run it again to finish."
              : review.error || "Unknown error"}>
      <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
        <button className="btn pri" onClick={async () => {
          try { await api.rerun(review.id); onRerun(); } catch (e) { setErr(String(e)); }
        }}>Run again</button>
        <button className="btn" onClick={async () => {
          await api.remove(review.id).catch(() => null); router.push("/reviews");
        }}>Delete</button>
      </div>
      {err && <p className="err">{err}</p>}
    </Notice>
  );
}

export function StatusBadge({ status }: { status: Review["status"] }) {
  const map = {
    done: ["grn", "Ready"], running: ["amb", "Analyzing"], queued: ["gry", "Queued"],
    error: ["red", "Failed"], interrupted: ["red", "Interrupted"],
  } as const;
  const [tone, label] = map[status];
  return <Badge tone={tone}>{label}</Badge>;
}
