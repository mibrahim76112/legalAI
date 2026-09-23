import Link from "next/link";
import Shell from "@/components/Shell";
import { PLAYBOOK_POSITIONS } from "@/lib/server/prompts";

export default function Playbooks() {
  const n = PLAYBOOK_POSITIONS.length;
  return (
    <Shell>
      <div className="phead">
        <div><h1>Playbooks</h1><div className="sub">Your standing positions</div></div>
        <span className="grow" />
        <Link href="/reviews/new" className="btn">⬆ Upload playbook</Link>
      </div>
      <div className="body-pad">
        <div className="tw">
          <table>
            <thead><tr><th>Playbook</th><th>Positions</th><th /></tr></thead>
            <tbody>
              <tr>
                <td>
                  <Link href="/playbooks/standard" className="fname">
                    <span className="fico">◈</span>
                    <b style={{ fontWeight: 580 }}>Standard playbook</b>
                  </Link>
                </td>
                <td style={{ color: "var(--tx2)" }}>{n}</td>
                <td style={{ textAlign: "right", color: "var(--tx3)" }}>→</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </Shell>
  );
}
