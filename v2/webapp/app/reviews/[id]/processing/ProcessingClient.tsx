"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { Review } from "@/lib/types";

export default function ProcessingClient({ review }: { review: Review }) {
  const router = useRouter();
  const stages = [
    "Reading document",
    "Checking playbook positions",
    "Identifying clauses",
    "Locating supporting language",
  ];
  const [i, setI] = useState(0);
  useEffect(() => {
    if (i >= stages.length) {
      const t = setTimeout(() => router.push(`/reviews/${review.id}`), 450);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setI((n) => n + 1), 1150);
    return () => clearTimeout(t);
  }, [i]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <div className="phead"><div><h1>Analyzing</h1>
        <div className="sub">{review.documentName}</div></div></div>
      <div className="body-pad narrow">
        <div className="card" style={{ padding: "16px 22px" }}>
          {stages.map((s, k) => (
            <div key={s} className={`stg ${k < i ? "done" : k === i ? "now" : ""}`}>
              <span className="ic">{k < i ? "✓" : ""}</span>{s}
            </div>
          ))}
          <div className="track"><i style={{ width: `${(i / stages.length) * 100}%` }} /></div>
        </div>
      </div>
    </>
  );
}
