import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { extractText } from "@/lib/server/extract";

export const maxDuration = 60;

const MAX_UPLOAD = 20 * 1024 * 1024;

export async function POST(req: Request) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const name = decodeURIComponent(req.headers.get("X-Filename") ?? "upload.txt");
  const data = new Uint8Array(await req.arrayBuffer());
  if (data.byteLength > MAX_UPLOAD) {
    return NextResponse.json({ error: "file too large (20 MB max)" }, { status: 400 });
  }
  try {
    return NextResponse.json({ text: await extractText(name, data) });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 400 });
  }
}
