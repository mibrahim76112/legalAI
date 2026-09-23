/**
 * One shared password for the whole app.
 *
 * The point is not to keep secrets: it is that a public URL with an HF token
 * behind it is an open door to a GPU endpoint billed by the hour. With
 * APP_PASSWORD unset (local dev) everything is open.
 */

import { cookies } from "next/headers";

export const COOKIE = "cr_auth";

async function digest(secret: string): Promise<string> {
  const bytes = new TextEncoder().encode(`contract-review:${secret}`);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Constant-time compare, so the cookie can't be guessed byte by byte. */
function sameString(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export const passwordRequired = () => Boolean(process.env.APP_PASSWORD);

export const tokenFor = (password: string) => digest(password);

export async function checkPassword(password: string): Promise<string | null> {
  const expected = process.env.APP_PASSWORD;
  if (!expected) return null;
  return sameString(password, expected) ? await tokenFor(expected) : null;
}

export async function authorized(): Promise<boolean> {
  if (!passwordRequired()) return true;
  const got = (await cookies()).get(COOKIE)?.value ?? "";
  return sameString(got, await tokenFor(process.env.APP_PASSWORD!));
}
