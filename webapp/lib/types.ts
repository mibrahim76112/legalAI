export type Tone = "red" | "amb" | "grn" | "gry";

export interface Span { text: string; start: number; end: number }

export interface Item {
  id: string;
  kind: "position" | "clause";
  title: string;          // the position sentence, or the clause category
  status: string;         // display label
  tone: Tone;
  evidence: Span[];
  note: string | null;    // assessment or risk note
  correct?: boolean;
}

export interface Review {
  id: string;
  documentName: string;
  representing: string;
  counterparty: string;
  playbookName: string;
  documentText: string;
  compliance: Item[];
  clauses: Item[];
}

export interface Meta {
  model: string;
  split: string;
  flagPrecision: number;
}
