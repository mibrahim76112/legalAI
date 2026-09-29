/**
 * Reviews and reviewer decisions.
 *
 * Vercel has no disk, so on a deploy this is Upstash Redis (Vercel KV). With
 * no Redis configured it falls back to JSON files under .data/, which keeps
 * `npm run dev` working with nothing to provision.
 */

import { promises as fs } from "fs";
import os from "os";
import path from "path";
import type { Decision, Progress, Review, ReviewSummary } from "../types";

const PREFIX = "review:";
const INDEX = "reviews";
// Without Redis, fall back to files. On a serverless host only the temp
// directory is writable and it does not survive between requests, so this is
// a development convenience, not storage: /api/health reports which is in use.
const ROOT = process.env.VERCEL ? path.join(os.tmpdir(), "contract-review")
                                : path.join(process.cwd(), ".data");
const DIR = path.join(ROOT, "reviews");
const ID_OK = /^[A-Za-z0-9_-]{1,64}$/;

function redis() {
  const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  // imported lazily so local dev never needs the package configured
  const { Redis } = require("@upstash/redis") as typeof import("@upstash/redis");
  return new Redis({ url, token });
}

export async function load(id: string): Promise<Review | null> {
  if (!ID_OK.test(id)) return null;
  const kv = redis();
  if (kv) return (await kv.get<Review>(PREFIX + id)) ?? null;
  try {
    return JSON.parse(await fs.readFile(path.join(DIR, `${id}.json`), "utf8")) as Review;
  } catch {
    return null;
  }
}

export async function save(rec: Review): Promise<Review> {
  if (!ID_OK.test(rec.id)) throw new Error("bad review id");
  const kv = redis();
  if (kv) {
    await kv.set(PREFIX + rec.id, rec);
    await kv.zadd(INDEX, { score: rec.createdAt, member: rec.id });
  } else {
    await fs.mkdir(DIR, { recursive: true });
    const p = path.join(DIR, `${rec.id}.json`);
    await fs.writeFile(`${p}.tmp`, JSON.stringify(rec));
    await fs.rename(`${p}.tmp`, p);   // atomic, so a crash can't truncate a review
  }
  return rec;
}

export async function update(id: string, fn: (r: Review) => void): Promise<Review | null> {
  const rec = await load(id);
  if (!rec) return null;
  fn(rec);
  rec.updatedAt = Date.now() / 1000;
  return save(rec);
}

export async function remove(id: string): Promise<boolean> {
  if (!ID_OK.test(id)) return false;
  const kv = redis();
  if (kv) {
    const n = await kv.del(PREFIX + id);
    await kv.zrem(INDEX, id);
    return n > 0;
  }
  try {
    await fs.unlink(path.join(DIR, `${id}.json`));
    return true;
  } catch {
    return false;
  }
}

export async function setDecision(id: string, itemId: string, decision: Decision, comment: string | null) {
  let found = false;
  const rec = await update(id, (r) => {
    for (const it of [...r.compliance, ...r.clauses]) {
      if (it.id === itemId) {
        it.decision = decision;
        it.comment = (comment ?? "").trim() || null;
        found = true;
      }
    }
  });
  if (rec && !found) throw new Error("no such finding");
  return rec;
}

export function summarize(r: Review): ReviewSummary {
  const items = [...(r.compliance ?? []), ...(r.clauses ?? [])];
  const tone = (t: string) => (r.compliance ?? []).filter((i) => i.tone === t).length;
  return {
    id: r.id, documentName: r.documentName, status: r.status, createdAt: r.createdAt,
    tasks: r.tasks, sample: r.sample ?? false, representing: r.representing,
    counts: { red: tone("red"), amb: tone("amb"), grn: tone("grn"),
              found: (r.clauses ?? []).filter((i) => i.tone === "grn").length },
    decided: items.filter((i) => i.decision).length,
    total: items.length,
  };
}

export async function list(): Promise<ReviewSummary[]> {
  const kv = redis();
  let recs: (Review | null)[];
  if (kv) {
    const ids = await kv.zrange<string[]>(INDEX, 0, -1, { rev: true });
    recs = ids.length ? await Promise.all(ids.map((i) => load(i))) : [];
  } else {
    let files: string[] = [];
    try { files = await fs.readdir(DIR); } catch { /* nothing saved yet */ }
    recs = await Promise.all(files.filter((f) => f.endsWith(".json")).map((f) => load(f.slice(0, -5))));
  }
  return recs.filter((r): r is Review => !!r)
    .sort((a, b) => b.createdAt - a.createdAt)
    .map(summarize);
}

/**
 * Progress lives in its own key, never inside the review.
 *
 * The run writes progress from one invocation while the page polls from
 * another. Merging it into the review record meant a stale in-flight copy
 * could overwrite the finished result, which is exactly what happened.
 */
const PROG = "progress:";
const PROG_DIR = path.join(ROOT, "progress");
const PROG_TTL = 600;

export async function setProgress(id: string, p: Progress | null): Promise<void> {
  if (!ID_OK.test(id)) return;
  const kv = redis();
  if (kv) {
    if (p) await kv.set(PROG + id, p, { ex: PROG_TTL });
    else await kv.del(PROG + id);
    return;
  }
  const f = path.join(PROG_DIR, `${id}.json`);
  if (!p) { await fs.unlink(f).catch(() => {}); return; }
  await fs.mkdir(PROG_DIR, { recursive: true });
  await fs.writeFile(f, JSON.stringify(p));
}

export async function getProgress(id: string): Promise<Progress | null> {
  if (!ID_OK.test(id)) return null;
  const kv = redis();
  if (kv) return (await kv.get<Progress>(PROG + id)) ?? null;
  try {
    return JSON.parse(await fs.readFile(path.join(PROG_DIR, `${id}.json`), "utf8")) as Progress;
  } catch {
    return null;
  }
}

/** A run dies with its serverless invocation; nothing resumes it. */
export const STALE_SECONDS = 180;

export const storageKind = () => (redis() ? "database" : "ephemeral");

export function markStale(r: Review): Review {
  if ((r.status === "running" || r.status === "queued") &&
      Date.now() / 1000 - (r.updatedAt ?? r.createdAt) > STALE_SECONDS) {
    return { ...r, status: "interrupted" };
  }
  return r;
}
