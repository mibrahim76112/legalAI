import { Suspense } from "react";
import Shell from "@/components/Shell";
import LiveClient from "./LiveClient";

// useSearchParams needs a Suspense boundary
export default function Page() {
  return <Shell><Suspense><LiveClient /></Suspense></Shell>;
}
