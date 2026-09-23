"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import ReviewFrame, { type FrameProps } from "@/components/ReviewFrame";
import Doc, { type Mark } from "@/components/Doc";
import { Badge } from "@/components/Badge";
import { DECISION_LABEL, triageOrder } from "@/lib/review";
import type { Decision } from "@/lib/types";

export default function Triage() {
  return (
    <ReviewFrame full>
      {(p) => <Suspense><Body {...p} /></Suspense>}
    </ReviewFrame>
  );
}

const ACTIONS: [Exclude<Decision, null>, string, string][] = [
  ["ok", "Accept", "A"], ["flag", "Flag for follow-up", "F"], ["skip", "Dismiss", "D"],
];

function Body({ review, decide }: FrameProps) {
  const router = useRouter();
  const order = useMemo(() => triageOrder(review), [review]);
  const want = useSearchParams().get("item");
  const idx = Math.max(0, order.findIndex((i) => i.id === want));
  const it = order[idx];
  const [ev, setEv] = useState(0);
  const [comment, setComment] = useState(it.comment ?? "");
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { setEv(0); setComment(it.comment ?? ""); setErr(null); setSaved(false); }, [it.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = (k: number) => {
    const n = order[(k + order.length) % order.length];
    router.replace(`/reviews/${review.id}/triage?item=${n.id}`, { scroll: false });
  };
  const nextUndecided = () => {
    for (let k = 1; k <= order.length; k++) {
      const n = order[(idx + k) % order.length];
      if (!n.decision && n.id !== it.id) return go((idx + k) % order.length);
    }
    go(idx + 1);
  };
  const save = async (d: Decision, c: string) => {
    setErr(null);
    try {
      await decide(it, d, c.trim() || null);
      setSaved(true);
      return true;
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      return false;
    }
  };
  const act = async (d: Exclude<Decision, null>) => {
    const nd = it.decision === d ? null : d;
    if ((await save(nd, comment)) && nd) nextUndecided();
  };

  // keyboard: J/K move, A/F/D decide; ignored while typing a comment
  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e) => {
    if (e.target instanceof HTMLTextAreaElement || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "j" || e.key === "ArrowDown") { e.preventDefault(); go(idx + 1); }
    else if (k === "k" || e.key === "ArrowUp") { e.preventDefault(); go(idx - 1); }
    else if (k === "a") act("ok");
    else if (k === "f") act("flag");
    else if (k === "d") act("skip");
    else if (k === "c") { e.preventDefault(); box.current?.focus(); }
  };
  useEffect(() => {
    const h = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const marks: Mark[] = it.evidence.map((s, i) => ({
    key: `${it.id}:${i}`, start: s.start, end: s.end, tone: it.tone, itemId: it.id,
  }));
  const n = it.evidence.length;
  const decided = order.filter((i) => i.decision).length;

  return (
    <div className="split">
      <div className="pane-l">
        <div className="tbar">
          <button className="btn sm" onClick={() => go(idx - 1)} title="Previous (K)">‹</button>
          <span className="muted">Finding {idx + 1} of {order.length} · {decided} decided</span>
          <button className="btn sm" onClick={() => go(idx + 1)} title="Next (J)">›</button>
        </div>
        <div className="tcard">
          <div className="kind">{it.kind === "position" ? "Playbook position" : "Clause category"}</div>
          <h2 className="ttl">{it.title}</h2>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
            <Badge tone={it.tone}>{it.status}</Badge>
            {it.decision && <span className="dec">{DECISION_LABEL[it.decision]}</span>}
          </div>

          {it.note && (
            <>
              <div className="lab">Why it matters</div>
              <p className="p">{it.note}</p>
            </>
          )}
          {it.assessment && <p className="muted" style={{ marginTop: 6 }}>{it.assessment}</p>}
          <div className="lab">Supporting language</div>
          {n === 0 ? (
            <p className="muted">
              {it.kind === "position" ? "Nothing in the contract addresses this position."
                : "No clause of this type was identified in this document."}
            </p>
          ) : (
            <>
              <div className={`q q-${it.tone}`}>{it.evidence[Math.min(ev, n - 1)].text}</div>
              {n > 1 && (
                <div className="qn">
                  <button onClick={() => setEv((ev - 1 + n) % n)}>‹</button>
                  <span>Passage {Math.min(ev, n - 1) + 1} of {n}</span>
                  <button onClick={() => setEv((ev + 1) % n)}>›</button>
                </div>
              )}
            </>
          )}

          <div className="lab">Your decision</div>
          <div className="acts">
            {ACTIONS.map(([d, l, k]) => (
              <button key={d} className={it.decision === d ? `on on-${d}` : ""} onClick={() => act(d)}>
                {l} <kbd>{k}</kbd>
              </button>
            ))}
          </div>
          <textarea ref={box} className="cmt" rows={3} value={comment}
                    placeholder="Add a comment (C). It's saved with the decision and appears in the report."
                    onChange={(e) => { setComment(e.target.value); setSaved(false); }}
                    onBlur={() => { if ((it.comment ?? "") !== comment.trim()) save(it.decision ?? null, comment); }} />
          <div className="savest">
            {err ? <span className="err">{err}</span>
              : saved ? <span className="muted">Saved</span>
              : <span className="muted">Keys: J/K next/previous · A/F/D decide · C comment</span>}
          </div>
          {decided === order.length && (
            <Link className="btn pri big" style={{ marginTop: 14 }} href={`/reviews/${review.id}/report`}>
              All findings decided: open the report →</Link>
          )}
        </div>
      </div>
      <div className="pane-r">
        <Doc text={review.documentText} marks={marks} current={n ? `${it.id}:${Math.min(ev, n - 1)}` : null} />
      </div>
    </div>
  );
}
