import { notFound } from "next/navigation";
import Shell from "@/components/Shell";
import ReviewView from "@/components/Review";
import { getReview, meta, reviews } from "@/lib/data";

export function generateStaticParams() { return reviews.map((r) => ({ id: r.id })); }

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const review = getReview(id);
  if (!review) notFound();
  return (
    <Shell>
      <ReviewView review={review} flagPrecision={meta.flagPrecision} />
    </Shell>
  );
}
