import Link from "next/link";

export default function Bar({ doc, who }: { doc?: string; who?: string }) {
  return (
    <div className="bar">
      <Link href="/reviews" style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="logo">CR</span>
        <span className="name">Contract Review</span>
      </Link>
      {doc && <><span className="sep">/</span><span style={{ color: "var(--ink2)" }}>{doc}</span></>}
      <span className="grow" />
      <Link href="/playbooks" style={{ fontSize: 13.5, color: "var(--ink2)" }}>Playbooks</Link>
      {who && (
        <span className="who"><span className="dot" />Acting for <b>{who}</b></span>
      )}
    </div>
  );
}
