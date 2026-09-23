import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { health } from "@/lib/server/hf";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  try {
    const { model } = await health();
    return NextResponse.json({ model: "ready", served: model, queued: 0 });
  } catch (e) {
    return NextResponse.json({ model: `failed: ${(e as Error).message}` });
  }
}
