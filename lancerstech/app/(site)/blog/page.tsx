import { Section } from "@/components/Section";
import { BlogList } from "@/components/BlogList";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "Blog - Lancers Tech" });

export default function BlogIndex() {
  return (
    <Section title="Blog" subtitle="Insights and updates">
      <BlogList />
    </Section>
  );
}