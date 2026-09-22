"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import ReviewView from "@/components/Review";
import { meta } from "@/lib/data";
import type { Review } from "@/lib/types";

interface Job {
  status: "queued" | "running" | "done" | "error";
  stage: string;
  done: number;
  total: number;
  window?: number;
  windows?: number;
  tasks: string[];
  documentName: string;
  review?: Review;
  error?: string;
}

const POLL_MS = 1500;

export default function LiveClient() {
  const id = useSearchParams().get("job");
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let stop = false;
    let t: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const r = await fetch(`/api/reviews/${id}`);
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "Review not found");
        if (stop) return;
        setJob(j); setErr(null);
        if (j.status !== "done" && j.status !== "error") t = setTimeout(tick, POLL_MS);
      } catch (e) {
        // the service restarting loses in-memory jobs; keep retrying briefly
        setErr(e instanceof Error ? e.message : String(e));
        if (!stop) t = setTimeout(tick, POLL_MS * 2);
      }
    };
    tick();
    return () => { stop = true; clearTimeout(t); };
  }, [id]);

  if (!id) return <Msg title="No review selected" body={<Link href="/reviews/new">Start a new review</Link>} />;
  if (job?.status === "done" && job.review)
    return <ReviewView review={job.review} flagPrecision={meta.flagPrecision} exportHref={null} />;
  if (job?.status === "error")
    return <Msg title="Review failed" body={job.error} />;

  const stages = [
    { k: "queued", l: "Reading document", on: true },
    { k: "positions", l: "Checking playbook positions", on: job?.tasks.includes("compliance") },
    { k: "clauses", l: "Identifying clauses", on: job?.tasks.includes("clauses") },
    { k: "notes", l: "Writing clause notes", on: job?.tasks.includes("clauses") },
  ].filter((s) => s.on);
  const cur = Math.max(0, stages.findIndex((s) => s.k === (job?.stage ?? "queued")));
  const pct = job && job.total ? (job.done / job.total) * 100 : 0;

  return (
    <>
      <div className="phead"><div><h1>Analyzing</h1>
        <div className="sub">{job?.documentName ?? "…"}</div></div></div>
      <div className="body-pad narrow">
        <div className="card" style={{ padding: "16px 22px" }}>
          {stages.map((s, k) => (
            <div key={s.k} className={`stg ${k < cur ? "done" : k === cur ? "now" : ""}`}>
              <span className="ic">{k < cur ? "✓" : ""}</span>{s.l}
              {k === cur && job && job.total > 0 && (
                <span className="muted" style={{ marginLeft: 8, fontSize: 12.5 }}>
                  {job.done}/{job.total}
                  {job.windows && job.windows > 1 && s.k !== "notes" && ` · window ${job.window} of ${job.windows}`}
                </span>
              )}
            </div>
          ))}
          <div className="track"><i style={{ width: `${pct}%` }} /></div>
        </div>
        {job?.status === "queued" && <p className="muted" style={{ marginTop: 12 }}>
          Waiting for the model (another review may be running, or it is still loading).</p>}
        {err && <p className="muted" style={{ marginTop: 12 }}>Reconnecting to the model service… ({err})</p>}
      </div>
    </>
  );
}

function Msg({ title, body }: { title: string; body: React.ReactNode }) {
  return (
    <>
      <div className="phead"><div><h1>{title}</h1></div></div>
      <div className="body-pad narrow"><p>{body}</p></div>
    </>
  );
}
