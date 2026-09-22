import type { Tone } from "@/lib/types";

export function Badge({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return <span className={`bg bg-${tone}`}><span className="d" />{children}</span>;
}

export function Tile({ tone, label, n }: { tone: Tone; label: string; n: number }) {
  return (
    <div className="tile">
      <div className="top"><span className={`d d-${tone}`} />{label}</div>
      <div className={`n n-${tone}`}>{n}</div>
    </div>
  );
}
