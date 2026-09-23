import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { runReview } from "@/lib/server/pipeline";
import { load, save, setProgress, update } from "@/lib/server/store";

// the whole review runs inside this one invocation; Vercel Hobby allows 60 s
export const maxDuration = 60;
export const dynamic = "force-dynamic";

export async function POST(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  const rec = await load(id);
  if (!rec) return NextResponse.json({ error: "no such review" }, { status: 404 });
  if (rec.status === "running") return NextResponse.json({ status: "running" });
  if (rec.status === "done") return NextResponse.json({ status: "done" });

  rec.status = "running";
  rec.error = null;
  await save(rec);
  await setProgress(id, { stage: "queued", done: 0, total: 0 });

  try {
    const out = await runReview(rec.documentText, {
      tasks: rec.tasks,
      positions: rec.positions ?? undefined,
      // progress goes to its own key so the polling GET (a different
      // invocation) can read it without racing this one's final write
      onProgress: (p) => setProgress(id, p),
    });
    await update(id, (r) => Object.assign(r, out, { status: "done" }));
    await setProgress(id, null);
    return NextResponse.json({ status: "done" });
  } catch (e) {
    const msg = (e as Error).message;
    await update(id, (r) => Object.assign(r, { status: "error", error: msg }));
    await setProgress(id, null);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
