"use client";

import { useMemo, useState } from "react";
import ReviewFrame, { type FrameProps } from "@/components/ReviewFrame";
import { download, reviewToMarkdown, slug } from "@/lib/report";
import { contractToMarkdown } from "@/lib/structure";
import { counts } from "@/lib/review";

export default function Report() {
  return <ReviewFrame>{(p) => <Body {...p} />}</ReviewFrame>;
}

function Body({ review }: FrameProps) {
  const md = useMemo(() => reviewToMarkdown(review), [review]);
  const [copied, setCopied] = useState(false);
  const c = counts(review);
  const name = slug(review.documentName);

  return (
    <div className="body-pad wide">
      <div className="next card">
        <div>
          <h2>Report</h2>
          <p className="muted" style={{ margin: "4px 0 0" }}>
            {c.decided < c.total
              ? `${c.total - c.decided} findings have no decision yet; they appear as "Not yet reviewed".`
              : "Every finding has a decision."} Markdown opens in any editor and pastes into email, Notion or Word.
          </p>
        </div>
        <span className="grow" />
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="btn pri" onClick={() => download(`${name}_review.md`, md)}>Download report (.md)</button>
          <button className="btn" onClick={async () => {
            await navigator.clipboard.writeText(md); setCopied(true); setTimeout(() => setCopied(false), 1500);
          }}>{copied ? "Copied" : "Copy"}</button>
          <button className="btn" onClick={() => download(`${name}.md`, contractToMarkdown(review.documentName, review.documentText))}>
            Contract as .md</button>
        </div>
      </div>
      <Markdown src={md} />
    </div>
  );
}

/** Renders the subset of Markdown that report.ts emits. */
function Markdown({ src }: { src: string }) {
  const out: React.ReactNode[] = [];
  const lines = src.split("\n");
  const inline = (s: string) => s.split(/(\*\*[^*]+\*\*|_[^_]+_)/g).map((t, i) =>
    t.startsWith("**") ? <b key={i}>{t.slice(2, -2)}</b>
      : t.startsWith("_") && t.endsWith("_") ? <i key={i}>{t.slice(1, -1)}</i> : t);
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) continue;
    if (l.startsWith("### ")) out.push(<h4 key={i}>{l.slice(4)}</h4>);
    else if (l.startsWith("## ")) out.push(<h3 key={i}>{l.slice(3)}</h3>);
    else if (l.startsWith("# ")) out.push(<h2 key={i} className="mdh1">{l.slice(2)}</h2>);
    else if (l.startsWith("> ")) out.push(<blockquote key={i}>{inline(l.slice(2))}</blockquote>);
    else if (l.startsWith("| ")) {
      const rows: string[][] = [];
      for (; i < lines.length && lines[i].startsWith("|"); i++) {
        if (!/^\|[-| ]+\|$/.test(lines[i])) rows.push(lines[i].split("|").slice(1, -1).map((x) => x.trim()));
      }
      i--;
      out.push(<table key={i} className="mdt"><tbody>
        {rows.slice(1).map((r, k) => <tr key={k}>{r.map((x, j) => <td key={j}>{x}</td>)}</tr>)}
      </tbody></table>);
    } else if (l.startsWith("- ")) {
      const items: string[] = [];
      for (; i < lines.length && lines[i].startsWith("- "); i++) items.push(lines[i].slice(2));
      i--;
      out.push(<ul key={i}>{items.map((x, k) => <li key={k}>{inline(x)}</li>)}</ul>);
    } else out.push(<p key={i}>{inline(l)}</p>);
  }
  return <div className="md card">{out}</div>;
}
