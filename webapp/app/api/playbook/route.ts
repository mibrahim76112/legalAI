import { NextResponse } from "next/server";
import { authorized } from "@/lib/server/auth";
import { extractText } from "@/lib/server/extract";
import { MAX_POSITIONS, parsePlaybook } from "@/lib/server/playbook";

export const maxDuration = 60;

const MAX_UPLOAD = 20 * 1024 * 1024;

export async function POST(req: Request) {
  if (!(await authorized())) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const name = decodeURIComponent(req.headers.get("X-Filename") ?? "playbook.txt");
  const data = new Uint8Array(await req.arrayBuffer());
  if (data.byteLength > MAX_UPLOAD) {
    return NextResponse.json({ error: "file too large (20 MB max)" }, { status: 400 });
  }
  try {
    const positions = parsePlaybook(await extractText(name, data));
    if (!positions.length) {
      return NextResponse.json({
        error: "No positions found in that file. Each position should be its own line or bullet.",
      }, { status: 400 });
    }
    return NextResponse.json({
      name: name.replace(/\.[^.]+$/, ""),
      positions: positions.slice(0, MAX_POSITIONS),
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 400 });
  }
}
