import { getSiteConfig } from "@/lib/content";
import { Section } from "@/components/Section";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "About - Lancers Tech" });

export default async function AboutPage() {
  const site = await getSiteConfig();
  return (
    <Section title={`About ${site.brandName}`} subtitle={site.description}>
      <div className="prose max-w-none">
        <p>{site.description || `${site.brandName} is a digital solutions company.`}</p>
      </div>
    </Section>
  );
}