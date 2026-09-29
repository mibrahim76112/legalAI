import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { list, save } from "@/lib/server/store";
import type { Review, Task } from "@/lib/types";

export const dynamic = "force-dynamic";

const TASKS: Task[] = ["compliance", "clauses"];
const newId = () => Math.random().toString(36).slice(2, 12);

export async function GET() {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.json(await list());
}

export async function POST(req: Request) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  const text = String(body.text ?? "").trim();
  const tasks = TASKS.filter((t) => (body.tasks ?? []).includes(t));
  const positions = Array.isArray(body.positions)
    ? body.positions.filter((p: unknown) => typeof p === "string" && p.trim()).slice(0, 60)
    : null;
  if (!text || !tasks.length) {
    return NextResponse.json({ error: "text and at least one check are required" }, { status: 400 });
  }
  const now = Date.now() / 1000;
  const rec: Review = {
    id: newId(), status: "queued", createdAt: now, updatedAt: now, tasks, sample: false,
    documentName: String(body.documentName || "Contract"), documentText: text,
    positions: positions?.length ? positions : null,
    representing: null, counterparty: null,
    playbookName: positions?.length ? String(body.playbookName || "Uploaded playbook") : "Standard playbook",
    compliance: [], clauses: [], stats: null,
  };
  await save(rec);
  // the work happens in POST /api/reviews/<id>/run, which the review page
  // starts: a serverless function cannot keep working after it responds
  return NextResponse.json({ id: rec.id });
}
