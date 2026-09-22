import Link from "next/link";
import Shell from "@/components/Shell";
import { Badge, Tile } from "@/components/Badge";
import { reviews, summary } from "@/lib/data";

export default function Reviews() {
  const t = reviews.reduce((a, r) => {
    const s = summary(r);
    return { red: a.red + s.red, amb: a.amb + s.amb, grn: a.grn + s.grn, n: a.n + s.total };
  }, { red: 0, amb: 0, grn: 0, n: 0 });

  return (
    <Shell>
      <div className="phead">
        <div>
          <h1>Reviews</h1>
          <div className="sub">{reviews.length} contracts · {t.n} findings</div>
        </div>
        <span className="grow" />
        <Link href="/reviews/new" className="btn pri">＋ New review</Link>
      </div>

      <div className="body-pad">
        <div className="tiles" style={{ marginBottom: 22 }}>
          <Tile tone="red" label="Need attention" n={t.red} />
          <Tile tone="amb" label="Not addressed" n={t.amb} />
          <Tile tone="grn" label="Meets standard" n={t.grn} />
          <Tile tone="gry" label="Contracts" n={reviews.length} />
        </div>

        <div className="tw">
          <table>
            <thead>
              <tr><th>Contract</th><th>Acting for</th><th>Findings</th><th /></tr>
            </thead>
            <tbody>
              {reviews.map((r) => {
                const s = summary(r);
                return (
                  <tr key={r.id}>
                    <td>
                      <Link href={`/reviews/${r.id}`} className="fname">
                        <span className="fico">▤</span>
                        <span>
                          <b style={{ fontWeight: 580 }}>{r.documentName}</b>
                          <div style={{ color: "var(--tx3)", fontSize: 12 }}>
                            {(r.documentText.length / 1000).toFixed(0)}k characters
                          </div>
                        </span>
                      </Link>
                    </td>
                    <td style={{ color: "var(--tx2)" }}>{r.representing}</td>
                    <td>
                      <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {s.red > 0 && <Badge tone="red">{s.red}</Badge>}
                        {s.amb > 0 && <Badge tone="amb">{s.amb}</Badge>}
                        {s.grn > 0 && <Badge tone="grn">{s.grn}</Badge>}
                      </span>
                    </td>
                    <td style={{ textAlign: "right", color: "var(--tx3)" }}>→</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Shell>
  );
}
