import { getFAQs } from "@/lib/content";
import { Section } from "@/components/Section";
import { FAQ } from "@/components/FAQ";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "FAQ - Lancers Tech" });

export default async function FAQPage() {
  const items = await getFAQs();
  return (
    <Section title="Frequently Asked Questions">
      <FAQ items={items} />
    </Section>
  );
}