import { Section } from "@/components/Section";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "Terms & Conditions - Lancers Tech" });

export default function TermsPage() {
  return (
    <Section title="Terms & Conditions">
      <div className="prose max-w-none opacity-80">Content will be populated from crawl.</div>
    </Section>
  );
}