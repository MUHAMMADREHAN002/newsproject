import { Section } from "@/components/Section";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "Courses - Lancers Tech" });

export default function CoursesPage() {
  return (
    <Section title="Courses & Training" subtitle="Learn with us">
      <div className="opacity-70">Content will be populated from crawl.</div>
    </Section>
  );
}