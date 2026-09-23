"use client";

import { useCallback, useEffect, useState } from "react";
import type { Decision, Item, Review, ReviewSummary, Task } from "./types";

// all calls go through next.config.js's /api rewrite to inference/server.py
async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`/api${path}`, init);
  } catch {
    throw new Error("Can't reach the review service. Please try again.");
  }
  if (r.status === 401 && typeof window !== "undefined") {
    window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
    throw new Error("Not signed in");
  }
  const j = await r.json().catch(() => null);
  if (!r.ok || j === null) {
    throw new Error(j?.error || (r.status >= 500
      ? "The review service is unavailable right now. Please try again."
      : `Request failed (${r.status})`));
  }
  return j as T;
}

const json = (method: string, body: unknown): RequestInit => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: () => call<{ model: string; queued: number }>("/health"),
  list: () => call<ReviewSummary[]>("/reviews"),
  get: (id: string) => call<Review>(`/reviews/${id}`),
  extract: (file: File) => call<{ text: string }>("/extract", {
    method: "POST", body: file, headers: { "X-Filename": encodeURIComponent(file.name) },
  }),
  playbook: (file: File) => call<{ name: string; positions: string[] }>("/playbook", {
    method: "POST", body: file, headers: { "X-Filename": encodeURIComponent(file.name) },
  }),
  submit: (text: string, tasks: Task[], documentName: string,
           positions?: string[], playbookName?: string) =>
    call<{ id: string }>("/reviews", json("POST",
      { text, tasks, documentName, positions, playbookName })),
  decide: (id: string, item: string, decision: Decision, comment: string | null) =>
    call<ReviewSummary>(`/reviews/${id}/items/${item}`, json("PUT", { decision, comment })),
  remove: (id: string) => call<object>(`/reviews/${id}`, { method: "DELETE" }),
  // the work runs inside this request; the page polls GET meanwhile
  run: (id: string) => call<{ status?: string }>(`/reviews/${id}/run`, { method: "POST" }),
  note: (id: string, item: string) =>
    call<{ note: string }>(`/reviews/${id}/items/${item}/note`, { method: "POST" }),
  login: (password: string) => call<{ ok: boolean }>("/login", json("POST", { password })),
};

const POLL_MS = 1500;

/** Loads a review and keeps polling while it is queued or running. */
export function useReview(id: string) {
  const [review, setReview] = useState<Review | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let stop = false;
    let t: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const r = await api.get(id);
        if (stop) return;
        setReview(r); setError(null);
        if (r.status === "queued" || r.status === "running") t = setTimeout(load, POLL_MS);
      } catch (e) {
        if (stop) return;
        setError(e instanceof Error ? e.message : String(e));
        t = setTimeout(load, POLL_MS * 2);
      }
    };
    load();
    return () => { stop = true; clearTimeout(t); };
  }, [id, tick]);

  const reload = useCallback(() => setTick((n) => n + 1), []);

  /** Optimistic local update of one finding, then save; rolls back on failure. */
  const decide = useCallback(async (item: Item, decision: Decision, comment: string | null) => {
    const patch = (d: Decision, c: string | null) => setReview((r) => r && ({
      ...r,
      compliance: r.compliance.map((i) => i.id === item.id ? { ...i, decision: d, comment: c } : i),
      clauses: r.clauses.map((i) => i.id === item.id ? { ...i, decision: d, comment: c } : i),
    }));
    const before = { d: item.decision ?? null, c: item.comment ?? null };
    patch(decision, comment);
    try {
      await api.decide(id, item.id, decision, comment);
    } catch (e) {
      patch(before.d, before.c);
      throw e;
    }
  }, [id]);

  const explain = useCallback(async (item: Item) => {
    const { note } = await api.note(id, item.id);
    const source = item.kind === "clause" ? "model" : "model-beta";
    setReview((r) => r && ({
      ...r,
      compliance: r.compliance.map((i) => i.id === item.id ? { ...i, note, noteSource: source } : i),
      clauses: r.clauses.map((i) => i.id === item.id ? { ...i, note, noteSource: source } : i),
    }));
  }, [id]);

  return { review, error, reload, decide, explain };
}
