export type Tone = "red" | "amb" | "grn" | "gry";
export type Decision = "ok" | "flag" | "skip" | null;
export type Task = "compliance" | "clauses";
export type Status = "queued" | "running" | "done" | "error" | "interrupted";

export interface Span { text: string; start: number; end: number }

export interface Item {
  id: string;
  kind: "position" | "clause";
  title: string;          // the position sentence, or the clause category
  status: string;         // display label
  tone: Tone;
  evidence: Span[];
  assessment?: string | null;  // fixed verdict sentence (positions)
  note: string | null;         // "why it matters"
  // model: task-3 note on a clause (in training); model-beta: task-3 note on a
  // position (outside training); fixed: canned text
  noteSource?: "model" | "model-beta" | "fixed" | null;
  decision?: Decision;
  comment?: string | null;
  correct?: boolean;
}

export interface Progress {
  stage: string;
  done: number;
  total: number;
  window?: number;
  windows?: number;
}

export interface Review {
  id: string;
  status: Status;
  createdAt: number;
  updatedAt?: number;
  tasks: Task[];
  sample?: boolean;
  error?: string | null;
  documentName: string;
  representing: string | null;
  counterparty: string | null;
  playbookName: string | null;
  documentText: string;
  compliance: Item[];
  clauses: Item[];
  stats?: { windows: number; unparsed: number; unlocated_quotes: number } | null;
  progress?: Progress | null;
}

export interface ReviewSummary {
  id: string;
  documentName: string;
  status: Status;
  createdAt: number;
  updatedAt?: number;
  tasks: Task[];
  sample: boolean;
  representing: string | null;
  counts: { red: number; amb: number; grn: number; found: number };
  decided: number;
  total: number;
}

export interface Meta {
  model: string;
  split: string;
  flagPrecision: number;
}
