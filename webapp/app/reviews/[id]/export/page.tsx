import Link from "next/link";
import { notFound } from "next/navigation";
import Shell from "@/components/Shell";
import { Tile } from "@/components/Badge";
import { getReview, reviews, summary } from "@/lib/data";

export function generateStaticParams() { return reviews.map((r) => ({ id: r.id })); }

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const r = getReview(id);
  if (!r) notFound();
  const s = summary(r);
  return (
    <Shell>
      <div className="phead">
        <div>
          <div className="crumb">
            <Link href={`/reviews/${r.id}`} style={{ color: "var(--tx3)" }}>{r.documentName}</Link> / Export
          </div>
          <h1>Export</h1>
        </div>
      </div>
      <div className="body-pad narrow">
        <div className="tiles" style={{ gridTemplateColumns: "repeat(3,1fr)", marginBottom: 22 }}>
          <Tile tone="red" label="Need attention" n={s.red} />
          <Tile tone="amb" label="Not addressed" n={s.amb} />
          <Tile tone="grn" label="Meets standard" n={s.grn} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[["▤", "Summary", "PDF"], ["≡", "Issues list", "DOCX"],
            ["✎", "Annotated contract", "DOCX"]].map(([ic, t, f]) => (
            <div key={t} className="card"
                 style={{ padding: "14px 16px", display: "flex", alignItems: "center", gap: 12 }}>
              <span className="fico">{ic}</span>
              <span style={{ flex: 1 }}>
                <b style={{ fontWeight: 560 }}>{t}</b>
                <div style={{ color: "var(--tx3)", fontSize: 12 }}>{f}</div>
              </span>
              <button className="btn sm">Download</button>
            </div>
          ))}
        </div>
      </div>
    </Shell>
  );
}
