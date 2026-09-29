import { NextResponse } from "next/server";
import { COOKIE, checkPassword, passwordRequired } from "@/lib/server/auth";

export async function POST(req: Request) {
  if (!passwordRequired()) return NextResponse.json({ ok: true });
  const { password } = await req.json().catch(() => ({ password: "" }));
  const token = await checkPassword(String(password ?? ""));
  if (!token) return NextResponse.json({ error: "Wrong password" }, { status: 401 });
  const res = NextResponse.json({ ok: true });
  res.cookies.set(COOKIE, token, {
    httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production",
    path: "/", maxAge: 60 * 60 * 24 * 7,
  });
  return res;
}
