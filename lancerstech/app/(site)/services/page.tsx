import { getAllServices } from "@/lib/content";
import { Section } from "@/components/Section";
import { ServiceCard } from "@/components/ServiceCard";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "Services - Lancers Tech" });

export default async function ServicesPage() {
  const services = await getAllServices();
  return (
    <Section title="Services" subtitle="Explore our offerings">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {services.map((s) => (
          <ServiceCard key={s.frontmatter.slug} title={s.frontmatter.title} description={s.frontmatter.description} href={`/services/${s.frontmatter.slug}`} icon={s.frontmatter.icon} />
        ))}
      </div>
    </Section>
  );
}