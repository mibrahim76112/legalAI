"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Shell from "@/components/Shell";
import { reviews } from "@/lib/data";

export default function New() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const pbRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<string | null>(null);
  const [pbFile, setPbFile] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [side, setSide] = useState("Our side");
  const [playbook, setPlaybook] = useState("standard");
  const [depth, setDepth] = useState("standard");

  return (
    <Shell>
      <div className="phead">
        <div><h1>New review</h1><div className="sub">Upload a contract to analyze</div></div>
      </div>

      <div className="body-pad narrow">
        <div className="field">
          <label>Contract</label>
          {file ? (
            <div className="picked">
              <span className="fico">▤</span>
              <span>
                <b style={{ fontWeight: 560 }}>{file}</b>
                <div style={{ color: "var(--tx3)", fontSize: 12 }}>Ready to analyze</div>
              </span>
              <button className="x" onClick={() => setFile(null)}>✕</button>
            </div>
          ) : (
            <div className={`up${drag ? " on" : ""}`}
                 onClick={() => fileRef.current?.click()}
                 onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                 onDragLeave={() => setDrag(false)}
                 onDrop={(e) => {
                   e.preventDefault(); setDrag(false);
                   const f = e.dataTransfer.files?.[0];
                   if (f) setFile(f.name);
                 }}>
              <div className="ic">⬆</div>
              <div className="t">Drop a contract here or browse</div>
              <div className="s">PDF, DOCX or TXT</div>
            </div>
          )}
          <input ref={fileRef} type="file" hidden accept=".pdf,.docx,.doc,.txt"
                 onChange={(e) => { const f = e.target.files?.[0]; if (f) setFile(f.name); }} />
        </div>

        <div className="field">
          <label>Playbook</label>
          <div className="h">Your standing positions. Upload your own or use the built-in set.</div>
          <select value={playbook} onChange={(e) => setPlaybook(e.target.value)}>
            <option value="standard">Standard playbook</option>
            {pbFile && <option value="custom">{pbFile}</option>}
          </select>
          {pbFile ? (
            <div className="picked" style={{ marginTop: 9 }}>
              <span className="fico">◈</span>
              <span><b style={{ fontWeight: 560 }}>{pbFile}</b></span>
              <button className="x" onClick={() => { setPbFile(null); setPlaybook("standard"); }}>✕</button>
            </div>
          ) : (
            <button className="btn sm" style={{ marginTop: 9 }}
                    onClick={() => pbRef.current?.click()}>
              ⬆ Upload playbook
            </button>
          )}
          <input ref={pbRef} type="file" hidden accept=".pdf,.docx,.doc,.txt,.csv"
                 onChange={(e) => {
                   const f = e.target.files?.[0];
                   if (f) { setPbFile(f.name); setPlaybook("custom"); }
                 }} />
        </div>

        <div className="field">
          <label>Acting for</label>
          <div className="opts">
            {["Our side", "Counterparty"].map((s) => (
              <button key={s} className={`opt${side === s ? " on" : ""}`}
                      onClick={() => setSide(s)}>{s}</button>
            ))}
          </div>
        </div>

        <div className="field">
          <label>Detail</label>
          <div className="seg">
            {["light", "standard", "thorough"].map((d) => (
              <button key={d} className={depth === d ? "on" : ""}
                      onClick={() => setDepth(d)}>
                {d[0].toUpperCase() + d.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <button className="btn pri big" disabled={!file}
                onClick={() => router.push(`/reviews/${reviews[0].id}/processing`)}>
          Analyze contract
        </button>
      </div>
    </Shell>
  );
}
