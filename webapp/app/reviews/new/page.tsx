"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Shell from "@/components/Shell";
import Stepper from "@/components/Stepper";
import Doc from "@/components/Doc";
import { api } from "@/lib/api";
import { CLAUSE_CATEGORIES, PLAYBOOK_POSITIONS } from "@/lib/server/prompts";
import type { Task } from "@/lib/types";

const STEPS = ["Upload", "Check the text", "Choose checks", "Analyze"];
// same budget as inference/pipeline.py: a window holds ~14,960 tokens
const WINDOW_TOKENS = 14960;
const POSITIONS = PLAYBOOK_POSITIONS;
const CATEGORIES = CLAUSE_CATEGORIES;

export default function New() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [name, setName] = useState("");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [book, setBook] = useState<{ name: string; positions: string[] } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const bookRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const stats = useMemo(() => {
    const tokens = Math.round(text.length / 3.6);  // ~3.6 chars per token in contracts
    const windows = Math.max(1, Math.ceil(tokens / WINDOW_TOKENS));
    const words = text ? text.trim().split(/\s+/).length : 0;
    return { tokens, windows, words };
  }, [text]);

  async function pick(f: File) {
    setFile(f); setErr(null); setBusy(true);
    setName(f.name.replace(/\.[^.]+$/, ""));
    try {
      const { text } = await api.extract(f);
      setText(text);
      // NDAs get playbook positions, commercial agreements get the clause list
      const nda = /confidential information|non-disclosure|receiving party/i.test(text.slice(0, 6000));
      setTasks(nda ? ["compliance"] : ["clauses"]);
      setStep(1);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setFile(null);
    } finally {
      setBusy(false);
    }
  }

  const positions = book?.positions ?? POSITIONS;
  const playbookLabel = book ? book.name : "standard";

  async function pickBook(f: File) {
    setErr(null); setBusy(true);
    try {
      setBook(await api.playbook(f));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function start() {
    setBusy(true); setErr(null);
    try {
      const { id } = await api.submit(text, tasks, name.trim() || "Contract",
                                      book?.positions, book?.name);
      // the review page starts the run and shows progress
      router.push(`/reviews/${id}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <Shell>
      <div className="phead">
        <div><h1>New review</h1>
          <div className="sub">Four steps: upload, check the text, choose checks, analyze</div></div>
      </div>
      <div className="body-pad narrow">
        <Stepper steps={STEPS} at={step} />

        {step === 0 && (
          <div className="field">
            <label>Contract</label>
            <div className="h">PDF, DOCX or TXT.</div>
            <div className={`up${drag ? " on" : ""}`} onClick={() => !busy && fileRef.current?.click()}
                 onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                 onDragLeave={() => setDrag(false)}
                 onDrop={(e) => { e.preventDefault(); setDrag(false);
                                  const f = e.dataTransfer.files?.[0]; if (f) pick(f); }}>
              <div className="ic">⬆</div>
              <div className="t">{busy ? "Reading the file…" : "Drop a contract here or browse"}</div>
              <div className="s">{file?.name ?? "PDF, DOCX or TXT"}</div>
            </div>
            <input ref={fileRef} type="file" hidden accept=".pdf,.docx,.txt"
                   onChange={(e) => { const f = e.target.files?.[0]; if (f) pick(f); }} />
          </div>
        )}

        {step === 1 && (
          <div className="field">
            <label>Does this look like your contract?</label>
            <div className="h">
              {stats.words.toLocaleString()} words · about {stats.tokens.toLocaleString()} tokens ·
              {stats.windows === 1 ? " fits in one pass" : ` split into ${stats.windows} overlapping parts`}
            </div>
            <div className="card prev"><Doc text={text} marks={[]} /></div>
            <p className="muted" style={{ marginTop: 10 }}>
              Only the text shown here is reviewed.
            </p>
            <div className="wnav">
              <button className="btn" onClick={() => { setStep(0); setFile(null); setText(""); }}>Back</button>
              <button className="btn pri" onClick={() => setStep(2)}>Text looks right →</button>
            </div>
          </div>
        )}

        {step === 2 && (
          <>
            <div className="field">
              <label>Name</label>
              <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="field">
              <label>Checks</label>
              <div className="h">Selected from the document; adjust if needed.</div>
              <div className="picks">
                <Pick on={tasks.includes("compliance")} title="Playbook positions"
                      sub="Is each standing position met, contradicted or missing?"
                      onClick={() => setTasks((t) => t.includes("compliance")
                        ? t.filter((x) => x !== "compliance") : [...t, "compliance"])} />
                <Pick on={tasks.includes("clauses")} title="Clause inventory"
                      sub={`${CATEGORIES.length} commercial clause types · which are present, with the text`}
                      onClick={() => setTasks((t) => t.includes("clauses")
                        ? t.filter((x) => x !== "clauses") : [...t, "clauses"])} />
              </div>
            </div>

            {tasks.includes("compliance") && (
              <div className="field">
                <label>Playbook</label>
                <div className="h">The standing positions each contract is checked against.</div>
                <div className="picks">
                  <Pick on={!book} title="Standard playbook"
                        sub={`${POSITIONS.length} positions for inbound NDAs`}
                        onClick={() => setBook(null)} />
                  <Pick on={!!book} title={book ? book.name : "Upload your playbook"}
                        sub={book ? `${book.positions.length} positions found`
                                  : "PDF, DOCX or TXT — one position per line or bullet"}
                        onClick={() => bookRef.current?.click()} />
                </div>
                <input ref={bookRef} type="file" hidden accept=".pdf,.docx,.txt"
                       onChange={(e) => { const f = e.target.files?.[0]; if (f) pickBook(f); }} />
                <details className="det2">
                  <summary>Positions being checked ({positions.length})</summary>
                  <ol>{positions.map((p) => <li key={p}>{p}</li>)}</ol>
                </details>
              </div>
            )}
            <div className="wnav">
              <button className="btn" onClick={() => setStep(1)}>Back</button>
              <button className="btn pri" disabled={!tasks.length} onClick={() => setStep(3)}>Continue →</button>
            </div>
          </>
        )}

        {step === 3 && (
          <div className="field">
            <label>Ready to analyze</label>
            <div className="card" style={{ padding: "16px 18px" }}>
              <Row k="Contract" v={`${name} · ${stats.words.toLocaleString()} words`} />
              <Row k="Checks" v={tasks.map((t) => t === "compliance"
                ? `${positions.length} playbook positions` : `${CATEGORIES.length} clause types`).join(" + ")} />
              <Row k="Playbook" v={positions.length + " positions · " + playbookLabel} />
            </div>
            <p className="muted" style={{ marginTop: 10 }}>
              The review is saved automatically when it finishes.
            </p>
            <div className="wnav">
              <button className="btn" onClick={() => setStep(2)}>Back</button>
              <button className="btn pri" disabled={busy} onClick={start}>
                {busy ? "Starting…" : "Analyze contract"}</button>
            </div>
          </div>
        )}

        {err && <p className="err">{err}</p>}
      </div>
    </Shell>
  );
}

function Pick({ on, title, sub, onClick }: { on: boolean; title: string; sub: string; onClick: () => void }) {
  return (
    <button className={`pick${on ? " on" : ""}`} onClick={onClick}>
      <span className="bx">{on ? "✓" : ""}</span>
      <span><b>{title}</b><span className="s">{sub}</span></span>
    </button>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return <div className="krow"><span className="muted">{k}</span><span>{v}</span></div>;
}
