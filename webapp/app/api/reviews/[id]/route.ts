import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { getProgress, load, markStale, remove } from "@/lib/server/store";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await params;
  const rec = await load(id);
  if (!rec) return NextResponse.json({ error: "no such review" }, { status: 404 });
  const out = markStale(rec);
  return NextResponse.json({ ...out, progress: out.status === "running" ? await getProgress(id) : null });
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.json({}, { status: await remove((await params).id) ? 200 : 404 });
}
