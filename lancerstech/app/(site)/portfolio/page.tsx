import { Section } from "@/components/Section";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "Portfolio - Lancers Tech" });

export default function PortfolioPage() {
  return (
    <Section title="Portfolio" subtitle="Selected work and case studies">
      <div className="opacity-70">Content will be populated from crawl.</div>
    </Section>
  );
}