"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import Doc from "./Doc";
import { Badge, Tile } from "./Badge";
import type { Item, Review, Tone } from "@/lib/types";

type Dec = "ok" | "flag" | "skip" | null;
const RANK: Record<Tone, number> = { red: 0, amb: 1, grn: 2, gry: 3 };

export default function ReviewView({ review, flagPrecision }:
  { review: Review; flagPrecision: number }) {
  const both = review.compliance.length > 0 && review.clauses.length > 0;
  const [tab, setTab] = useState<"compliance" | "clauses">(
    review.compliance.length ? "compliance" : "clauses"
  );
  const [filter, setFilter] = useState<Tone | "all">("all");
  const [open, setOpen] = useState<string | null>(null);
  const [cur, setCur] = useState(0);
  const [dec, setDec] = useState<Record<string, Dec>>({});

  const src = tab === "compliance" ? review.compliance : review.clauses;
  const list = useMemo(
    () => src.filter((i) => filter === "all" || i.tone === filter)
             .slice().sort((a, b) => RANK[a.tone] - RANK[b.tone]),
    [src, filter]
  );
  const sel = src.find((i) => i.id === open) ?? null;
  const n = (t: Tone) => src.filter((i) => i.tone === t).length;

  const all = [...review.compliance, ...review.clauses];
  const tot = {
    red: all.filter((i) => i.tone === "red").length,
    amb: all.filter((i) => i.tone === "amb").length,
    grn: all.filter((i) => i.tone === "grn").length,
  };
  const reviewed = Object.values(dec).filter(Boolean).length;

  const FILTERS: { k: Tone | "all"; l: string }[] = [
    { k: "all", l: "All" },
    { k: "red", l: tab === "compliance" ? "Needs attention" : "Detected" },
    { k: "amb", l: "Not addressed" },
    { k: "grn", l: tab === "compliance" ? "Meets standard" : "Found" },
    { k: "gry", l: "Not detected" },
  ];

  return (
    <>
      <div className="phead">
        <div>
          <div className="crumb">
            <Link href="/reviews" style={{ color: "var(--tx3)" }}>Reviews</Link> / Review
          </div>
          <h1>{review.documentName}</h1>
          <div className="sub">
            Acting for {review.representing} · {review.playbookName}
          </div>
        </div>
        <span className="grow" />
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {tot.red > 0 && <Badge tone="red">{tot.red} need attention</Badge>}
          <span className="muted" style={{ fontSize: 12.5 }}>
            {reviewed}/{all.length} reviewed
          </span>
          <Link href={`/reviews/${review.id}/export`} className="btn">Export</Link>
        </div>
      </div>

      <div className="split">
        <div className="pane-l">
          <div className="seg2">
            <button className={tab === "compliance" ? "on" : ""}
                    disabled={!review.compliance.length}
                    onClick={() => { setTab("compliance"); setOpen(null); setFilter("all"); }}>
              Compliance {review.compliance.length > 0 && `· ${review.compliance.length}`}
            </button>
            <button className={tab === "clauses" ? "on" : ""}
                    disabled={!review.clauses.length}
                    onClick={() => { setTab("clauses"); setOpen(null); setFilter("all"); }}>
              Clauses {review.clauses.length > 0 && `· ${review.clauses.length}`}
            </button>
          </div>

          <div className="filters">
            {FILTERS.filter((f) => f.k === "all" || n(f.k as Tone) > 0).map((f) => (
              <button key={f.k} className={`f${filter === f.k ? " on" : ""}`}
                      onClick={() => setFilter(f.k)}>
                {f.k !== "all" && <span className={`d d-${f.k}`} />}
                {f.l}
                <span className="c">{f.k === "all" ? src.length : n(f.k as Tone)}</span>
              </button>
            ))}
          </div>

          <div className="rows">
            {list.map((it) => {
              const on = it.id === open;
              return (
                <div key={it.id} className={`row${on ? " on" : ""}`}>
                  <button className="hit" onClick={() => { setOpen(on ? null : it.id); setCur(0); }}>
                    <span className={`sd d-${it.tone}`} />
                    <span className="tt">
                      <div className="tl">{it.title}</div>
                      <div className="mt">
                        <span>{it.status}</span>
                        {it.evidence.length > 0 && <span>· {it.evidence.length} passage{it.evidence.length > 1 ? "s" : ""}</span>}
                        {dec[it.id] && (
                          <Badge tone={dec[it.id] === "ok" ? "grn" : dec[it.id] === "flag" ? "amb" : "gry"}>
                            {dec[it.id] === "ok" ? "accepted" : dec[it.id] === "flag" ? "flagged" : "dismissed"}
                          </Badge>
                        )}
                      </div>
                    </span>
                    <span className="chev">{on ? "▲" : "▼"}</span>
                  </button>
                  {on && <Detail item={it} cur={cur} setCur={setCur}
                                dec={dec[it.id] ?? null}
                                setDec={(d) => setDec((p) => ({ ...p, [it.id]: d }))}
                                flagPrecision={flagPrecision} />}
                </div>
              );
            })}
            {list.length === 0 && (
              <div className="empty"><div className="ic">◌</div>Nothing in this filter</div>
            )}
          </div>
        </div>

        <div className="pane-r">
          <Doc text={review.documentText}
               spans={sel ? sel.evidence : []}
               tone={sel ? sel.tone : "gry"} cur={cur}
               title={sel ? sel.title : null} />
        </div>
      </div>
    </>
  );
}

function Detail({ item, cur, setCur, dec, setDec, flagPrecision }: {
  item: Item; cur: number; setCur: (n: number) => void; dec: Dec;
  setDec: (d: Dec) => void; flagPrecision: number;
}) {
  const n = item.evidence.length;
  const i = Math.min(cur, Math.max(n - 1, 0));
  return (
    <div className="det">
      <div className="lab">Supporting language</div>
      {n === 0 ? (
        <p className="muted">Nothing found in the document for this.</p>
      ) : (
        <>
          <div className={`q q-${item.tone}`}>{item.evidence[i].text}</div>
          <div className="qn">
            {n > 1 && (
              <>
                <button onClick={() => setCur((i - 1 + n) % n)}>‹</button>
                <span>{i + 1} of {n}</span>
                <button onClick={() => setCur((i + 1) % n)}>›</button>
              </>
            )}
            <button className="go" onClick={() => setCur(i)}>↗ Show in document</button>
          </div>
        </>
      )}

      {item.note && (
        <>
          <div className="lab">{item.kind === "position" ? "Assessment" : "Why it matters"}</div>
          <p className="p">{item.note}</p>
        </>
      )}

      <div className="acts">
        {([["ok", "Accept"], ["flag", "Flag"], ["skip", "Dismiss"]] as [Dec, string][])
          .map(([d, l]) => (
            <button key={d} className={dec === d ? "on" : ""}
                    onClick={() => setDec(dec === d ? null : d)}>{l}</button>
          ))}
      </div>
    </div>
  );
}
