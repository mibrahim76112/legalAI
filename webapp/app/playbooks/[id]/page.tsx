import Link from "next/link";
import Shell from "@/components/Shell";
import { PLAYBOOK_POSITIONS } from "@/lib/server/prompts";

export function generateStaticParams() { return [{ id: "standard" }]; }

export default async function Page() {
  const items = PLAYBOOK_POSITIONS;
  return (
    <Shell>
      <div className="phead">
        <div>
          <div className="crumb">
            <Link href="/playbooks" style={{ color: "var(--tx3)" }}>Playbooks</Link> / Standard
          </div>
          <h1>Standard playbook</h1>
          <div className="sub">{items.length} positions</div>
        </div>
        <span className="grow" />
        <button className="btn">＋ Add position</button>
      </div>
      <div className="body-pad">
        <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 760 }}>
          {items.map((title, i) => (
            <div key={title} className="card"
                 style={{ padding: "14px 18px", display: "flex", gap: 14 }}>
              <span style={{ color: "var(--tx3)", fontSize: 12.5, fontWeight: 600,
                             minWidth: 20 }}>{i + 1}</span>
              <span style={{ fontSize: 13.8, lineHeight: 1.5 }}>{title}</span>
            </div>
          ))}
        </div>
      </div>
    </Shell>
  );
}
