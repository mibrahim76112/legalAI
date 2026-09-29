"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/reviews", ic: "◳", label: "Reviews" },
  { href: "/playbooks", ic: "◈", label: "Playbooks" },
];

export default function Rail() {
  const path = usePathname();
  return (
    <aside className="rail">
      <Link href="/reviews" className="mark">
        <span className="sq">CR</span>
        <span className="t">ContractReview</span>
      </Link>

      <div className="nav">
        <Link href="/reviews/new" className="on" style={{ marginBottom: 6 }}>
          <span className="ic">＋</span><span>New review</span>
        </Link>
        <div className="grp">Workspace</div>
        {NAV.map((n) => (
          <Link key={n.href} href={n.href}
                className={path.startsWith(n.href) && !path.includes("/new") ? "on" : ""}>
            <span className="ic">{n.ic}</span><span>{n.label}</span>
          </Link>
        ))}
      </div>

      <div className="foot">
        <div className="u"><span className="av">A</span><span>Review workspace</span></div>
      </div>
    </aside>
  );
}
