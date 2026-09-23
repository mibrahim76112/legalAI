import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { writeNote } from "@/lib/server/pipeline";
import { load, update } from "@/lib/server/store";

export const maxDuration = 60;
export const dynamic = "force-dynamic";

/** Write (or rewrite) one finding's note, on request from the review page. */
export async function POST(_req: Request, { params }: { params: Promise<{ id: string; item: string }> }) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id, item } = await params;
  const rec = await load(id);
  if (!rec) return NextResponse.json({ error: "no such review" }, { status: 404 });
  const found = [...rec.compliance, ...rec.clauses].find((i) => i.id === item);
  if (!found) return NextResponse.json({ error: "no such finding" }, { status: 404 });

  try {
    const note = await writeNote(found.title, found.evidence.map((s) => s.text));
    // clause notes are what task 3 was trained on; a playbook position passed as
    // the category is outside that, and the review keeps track of which is which
    const noteSource = found.kind === "clause" ? "model" : "model-beta";
    await update(id, (r) => {
      for (const it of [...r.compliance, ...r.clauses]) {
        if (it.id === item) { it.note = note; it.noteSource = noteSource; }
      }
    });
    return NextResponse.json({ note });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 400 });
  }
}
