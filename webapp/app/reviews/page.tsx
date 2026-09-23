"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import Shell from "@/components/Shell";
import { Badge, Tile } from "@/components/Badge";
import { StatusBadge } from "@/components/ReviewFrame";
import { api } from "@/lib/api";
import type { ReviewSummary } from "@/lib/types";

export default function Reviews() {
  const [rows, setRows] = useState<ReviewSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    api.list().then((r) => { setRows(r); setErr(null); })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    load();
    // a review running in another tab finishes without us; refresh while any is busy
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [load]);

  const busy = rows?.some((r) => r.status === "queued" || r.status === "running");
  const t = (rows ?? []).filter((r) => r.status === "done")
    .reduce((a, r) => ({ red: a.red + r.counts.red, amb: a.amb + r.counts.amb,
                         grn: a.grn + r.counts.grn, open: a.open + (r.total - r.decided) }),
            { red: 0, amb: 0, grn: 0, open: 0 });

  return (
    <Shell>
      <div className="phead">
        <div>
          <h1>Reviews</h1>
          <div className="sub">
            {rows ? `${rows.length} contract${rows.length === 1 ? "" : "s"}` : "Loading…"}
            {busy && " · one is being analyzed"}
          </div>
        </div>
        <span className="grow" />
        <Link href="/reviews/new" className="btn pri">＋ New review</Link>
      </div>

      <div className="body-pad">
        {err && <p className="err" style={{ marginBottom: 16 }}>{err}</p>}

        {rows && rows.length > 0 && (
          <div className="tiles" style={{ marginBottom: 22 }}>
            <Tile tone="red" label="Needs attention" n={t.red} />
            <Tile tone="amb" label="Not addressed" n={t.amb} />
            <Tile tone="grn" label="Meets standard" n={t.grn} />
            <Tile tone="gry" label="Awaiting your decision" n={t.open} />
          </div>
        )}

        {rows?.length === 0 && (
          <div className="card empty">
            <div className="ic">▤</div>
            No reviews yet. <Link href="/reviews/new" style={{ color: "var(--acc)" }}>Upload a contract</Link> to start.
          </div>
        )}

        {rows && rows.length > 0 && (
          <div className="tw">
            <table>
              <thead>
                <tr><th>Contract</th><th>Status</th><th>Findings</th><th>Your progress</th><th /></tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <Link href={`/reviews/${r.id}`} className="fname">
                        <span className="fico">▤</span>
                        <span>
                          <b style={{ fontWeight: 580 }}>{r.documentName}</b>
                          <div style={{ color: "var(--tx3)", fontSize: 12 }}>
                            {r.sample ? "Sample" : new Date(r.createdAt * 1000).toLocaleString()}
                          </div>
                        </span>
                      </Link>
                    </td>
                    <td><StatusBadge status={r.status} /></td>
                    <td>
                      <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {r.counts.red > 0 && <Badge tone="red">{r.counts.red}</Badge>}
                        {r.counts.amb > 0 && <Badge tone="amb">{r.counts.amb}</Badge>}
                        {r.counts.grn > 0 && <Badge tone="grn">{r.counts.grn}</Badge>}
                        {r.counts.found > 0 && <Badge tone="gry">{r.counts.found} clauses</Badge>}
                      </span>
                    </td>
                    <td style={{ color: "var(--tx2)", whiteSpace: "nowrap" }}>
                      {r.status === "done" ? `${r.decided} of ${r.total}` : "—"}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button className="del" title="Delete this review"
                              onClick={async () => {
                                if (!confirm(`Delete the review of ${r.documentName}?`)) return;
                                try { await api.remove(r.id); load(); }
                                catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
                              }}>✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Shell>
  );
}
