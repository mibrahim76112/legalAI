import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ContractReview",
  description: "Contract review with evidence anchoring",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
