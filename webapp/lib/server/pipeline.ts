/**
 * Contract -> Review, against the served model. Port of inference/pipeline.py.
 *
 * Questions run concurrently: the server answers many at once and caches the
 * shared contract prefix itself, which is what turns a 230 s local review into
 * ~17 s. Windows are still the outer loop so a long contract is covered.
 */

import { ask, countTokens, PROMPT_BUDGET, type Msg } from "./hf";
import {
  CLAUSE_CATEGORIES, PLAYBOOK_POSITIONS, POSITION_STATUS, SILENT_NOTE,
  TASK1_SYSTEM, TASK2_SYSTEM, TASK3_SYSTEM, task1User, task2User, task3User,
} from "./prompts";
import { norm, spansOf } from "./spans";
import type { Item, Progress, Task, Tone } from "../types";

const PARALLEL = 8;
// leaves room for the system message, the question and the template tokens
const PROMPT_OVERHEAD = 400;
const WINDOW_TOKENS = PROMPT_BUDGET - PROMPT_OVERHEAD;
const OVERLAP_CHARS = 4000;
const SNAP_CHARS = 400;
// conservative: contracts run ~3.6 chars/token, so this over-estimates tokens
const CHARS_PER_TOKEN = 3.2;

// Across windows a conflict anywhere outranks support anywhere, which outranks
// silence; a clause only has to appear in one window to count.
const VERDICT_RANK: Record<string, number> = { Contradiction: 0, Entailment: 1, NotMentioned: 2 };

/** Run fn over items with at most `limit` in flight, results in order. */
async function pool<T, R>(items: T[], limit: number, fn: (t: T, i: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    for (let i = next++; i < items.length; i = next++) out[i] = await fn(items[i], i);
  }));
  return out;
}

function stripWrappers(raw: string): string {
  return (raw || "")
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/^[\s\S]*?<\/think>/i, (m) => (/<\/think>/i.test(m) ? " " : m))
    .replace(/```(?:json)?/gi, " ")
    .trim();
}

/** First balanced JSON object in the text, like doc_harness._find_json. */
function findJson(text: string): Record<string, unknown> | null {
  try {
    const o = JSON.parse(text);
    if (o && typeof o === "object") return o as Record<string, unknown>;
  } catch { /* fall through to the brace scan */ }
  let depth = 0, start = -1;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "{") { if (depth === 0) start = i; depth++; }
    else if (text[i] === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        try {
          const o = JSON.parse(text.slice(start, i + 1));
          if (o && typeof o === "object") return o as Record<string, unknown>;
        } catch { start = -1; }
      }
    }
  }
  return null;
}

const evidenceOf = (o: Record<string, unknown> | null): string[] => {
  const ev = o?.evidence;
  const list = typeof ev === "string" ? [ev] : Array.isArray(ev) ? ev : [];
  return list.filter((e): e is string => typeof e === "string" && e.trim().length > 0);
};

const VERDICTS = ["Entailment", "Contradiction", "NotMentioned"];

function parseVerdict(raw: string): [string | null, string[]] {
  const o = findJson(stripWrappers(raw));
  const v = typeof o?.verdict === "string"
    ? VERDICTS.find((c) => c.toLowerCase() === (o.verdict as string).replace(/[\s_-]/g, "").toLowerCase())
    : undefined;
  return [v ?? null, evidenceOf(o)];
}

function parsePresent(raw: string): [boolean | null, string[]] {
  const o = findJson(stripWrappers(raw));
  return [typeof o?.present === "boolean" ? o.present : null, evidenceOf(o)];
}

/** Shrink [start,end) until the server's tokenizer says it fits the window. */
async function fit(text: string, start: number, end: number): Promise<number> {
  for (let i = 0; i < 6; i++) {
    const n = await countTokens(text.slice(start, end));
    if (n === null || n <= WINDOW_TOKENS) return end;
    // scale by the measured ratio, with a little margin, then snap to a line
    const target = Math.floor((end - start) * (WINDOW_TOKENS / n) * 0.95);
    const shrunk = start + Math.max(target, 1000);
    if (shrunk >= end) return end;
    const nl = text.lastIndexOf("\n", shrunk);
    end = nl > start + SNAP_CHARS ? nl + 1 : shrunk;
  }
  return end;
}

/**
 * Character ranges covering the text, each within the model's window.
 *
 * The character estimate only picks the starting size: a token-dense contract
 * can still exceed the window, and the endpoint rejects the whole request when
 * it does, so every window is checked against the real tokenizer.
 */
export async function windows(text: string): Promise<[number, number][]> {
  if (text.length / CHARS_PER_TOKEN <= WINDOW_TOKENS) {
    const n = await countTokens(text);
    if (n === null || n <= WINDOW_TOKENS) return [[0, text.length]];
  }
  const span = Math.floor(WINDOW_TOKENS * CHARS_PER_TOKEN);
  const out: [number, number][] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + span, text.length);
    if (end < text.length) {
      const nl = text.lastIndexOf("\n", end);
      if (nl > start + SNAP_CHARS) end = nl + 1;
    }
    end = await fit(text, start, end);
    out.push([start, end]);
    if (end >= text.length) break;
    start = Math.max(end - OVERLAP_CHARS, start + 1);
  }
  return out;
}

export interface RunOptions {
  tasks: Task[];
  positions?: string[];          // an uploaded playbook, else the standard one
  onProgress?: (p: Progress) => void | Promise<void>;
}

export interface RunResult {
  compliance: Item[];
  clauses: Item[];
  stats: { windows: number; unparsed: number; unlocated_quotes: number };
  representing: string;
  counterparty: string;
  playbookName: string;
}

export async function runReview(text: string, { tasks, positions, onProgress }: RunOptions): Promise<RunResult> {
  const playbook = positions?.length ? positions : PLAYBOOK_POSITIONS;
  const wins = await windows(text);
  const qs: { kind: "nli" | "cuad"; q: string }[] = [
    ...(tasks.includes("compliance") ? playbook.map((q) => ({ kind: "nli" as const, q })) : []),
    ...(tasks.includes("clauses") ? CLAUSE_CATEGORIES.map((q) => ({ kind: "cuad" as const, q })) : []),
  ];
  const stats = { windows: wins.length, unparsed: 0, unlocated_quotes: 0 };
  const total = wins.length * qs.length;
  let done = 0;

  const perQ = new Map<string, [string | boolean | null, string[]][]>();
  for (const [w, [a, b]] of wins.entries()) {
    const chunk = text.slice(a, b);
    const answers = await pool(qs, PARALLEL, async ({ kind, q }) => {
      const msgs: Msg[] = kind === "nli"
        ? [{ role: "system", content: TASK1_SYSTEM }, { role: "user", content: task1User(chunk, q) }]
        : [{ role: "system", content: TASK2_SYSTEM }, { role: "user", content: task2User(chunk, q) }];
      const raw = await ask(msgs);
      const parsed = kind === "nli" ? parseVerdict(raw) : parsePresent(raw);
      if (parsed[0] === null) stats.unparsed++;
      done++;
      if (done % 4 === 0 || done === total) {
        await onProgress?.({ stage: kind === "cuad" ? "clauses" : "positions", done, total,
                             window: w + 1, windows: wins.length });
      }
      return parsed;
    });
    qs.forEach(({ kind, q }, i) => {
      const key = `${kind}:${q}`;
      perQ.set(key, [...(perQ.get(key) ?? []), answers[i]]);
    });
  }

  const located = (ev: string[]) => {
    const { spans, dropped } = spansOf(text, ev);
    stats.unlocated_quotes += dropped;
    return spans;
  };

  const compliance: Item[] = [];
  if (tasks.includes("compliance")) {
    playbook.forEach((q, i) => {
      const res = (perQ.get(`nli:${q}`) ?? []).filter(([v]) => v) as [string, string[]][];
      const verdict = res.map(([v]) => v).sort((a, b) => VERDICT_RANK[a] - VERDICT_RANK[b])[0] ?? "NotMentioned";
      const ev = res.filter(([v]) => v === verdict).flatMap(([, e]) => e);
      const [status, tone, assessment] = POSITION_STATUS[verdict as keyof typeof POSITION_STATUS];
      const silent = verdict === "NotMentioned";
      compliance.push({
        id: `p${i}`, kind: "position", title: q, status, tone: tone as Tone,
        evidence: located(ev), assessment,
        note: silent ? SILENT_NOTE : null, noteSource: silent ? "fixed" : null,
      });
    });
  }

  const clauses: Item[] = [];
  if (tasks.includes("clauses")) {
    CLAUSE_CATEGORIES.forEach((c, i) => {
      const res = perQ.get(`cuad:${c}`) ?? [];
      const present = res.some(([p]) => p === true);
      const ev = res.filter(([p]) => p === true).flatMap(([, e]) => e);
      clauses.push({
        id: `c${i}`, kind: "clause", title: c,
        status: present ? "Found" : "Not detected", tone: present ? "grn" : "gry",
        evidence: present ? located(ev) : [], note: null, noteSource: null,
      });
    });
  }

  return {
    compliance, clauses, stats,
    representing: tasks.includes("compliance") ? "Receiving Party" : "Counterparty",
    counterparty: tasks.includes("compliance") ? "Disclosing Party" : "—",
    playbookName: positions?.length ? "Uploaded playbook" : "Standard playbook",
  };
}

export { norm };

/**
 * One finding's "why it matters" note, written on request.
 *
 * Task 3 was trained on (clause category, clause text), so it needs the quoted
 * evidence: a finding with nothing quoted has nothing to explain.
 */
export async function writeNote(title: string, evidence: string[]): Promise<string> {
  if (!evidence.length) throw new Error("This finding has no supporting language to explain.");
  const raw = await ask([{ role: "system", content: TASK3_SYSTEM },
                         { role: "user", content: task3User(title, evidence.join(" ")) }]);
  return stripWrappers(raw).trim();
}
