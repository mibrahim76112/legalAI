import raw from "@/data/ui.json";
import type { Meta, Review } from "./types";

const p = raw as unknown as { meta: Meta; reviews: Review[] };
export const meta = p.meta;
export const reviews = p.reviews;
export const getReview = (id: string) => reviews.find((r) => r.id === id);

export function summary(r: Review) {
  const all = [...r.compliance, ...r.clauses];
  return {
    red: all.filter((i) => i.tone === "red").length,
    amb: all.filter((i) => i.tone === "amb").length,
    grn: all.filter((i) => i.tone === "grn").length,
    total: all.length,
  };
}
