import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { setDecision, summarize } from "@/lib/server/store";
import type { Decision } from "@/lib/types";

const DECISIONS = ["ok", "flag", "skip", null];

export async function PUT(req: Request, { params }: { params: Promise<{ id: string; item: string }> }) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id, item } = await params;
  const body = await req.json().catch(() => ({}));
  const decision = (body.decision ?? null) as Decision;
  if (!DECISIONS.includes(decision)) {
    return NextResponse.json({ error: "decision must be one of ok, flag, skip or null" }, { status: 400 });
  }
  try {
    const rec = await setDecision(id, item, decision, body.comment ?? null);
    if (!rec) return NextResponse.json({ error: "no such review" }, { status: 404 });
    return NextResponse.json(summarize(rec));
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 404 });
  }
}
