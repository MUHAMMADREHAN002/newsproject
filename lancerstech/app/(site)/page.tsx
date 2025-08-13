import { getSiteConfig, getAllServices, getFAQs } from "@/lib/content";
import { Hero } from "@/components/Hero";
import { Section } from "@/components/Section";
import { ServiceCard } from "@/components/ServiceCard";
import { FAQ } from "@/components/FAQ";
import NewsletterForm from "@/components/NewsletterForm";

export default async function HomePage() {
  const site = await getSiteConfig();
  const services = await getAllServices();
  const faqs = await getFAQs();

  return (
    <div>
      <Hero title={site.hero?.title || site.tagline || site.brandName} subtitle={site.hero?.subtitle || site.description} ctaLabel={site.hero?.ctaLabel} ctaHref={site.hero?.ctaHref} />

      {services.length > 0 && (
        <Section title="Services" subtitle="What we can do for you">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {services.map((s) => (
              <ServiceCard key={s.frontmatter.slug} title={s.frontmatter.title} description={s.frontmatter.description} href={`/services/${s.frontmatter.slug}`} icon={s.frontmatter.icon} />
            ))}
          </div>
        </Section>
      )}

      {faqs.length > 0 && (
        <Section title="FAQ" subtitle="Answers to common questions">
          <FAQ items={faqs} />
        </Section>
      )}

      <Section title="Stay in the loop" subtitle="Get updates occasionally">
        <NewsletterForm />
      </Section>

      {site.whatsappNumber && (
        <a
          href={`https://wa.me/${site.whatsappNumber}`}
          className="fixed bottom-5 right-5 bg-green-500 text-white rounded-full px-4 py-3 shadow-lg hover:opacity-90"
          aria-label="WhatsApp Chat"
          target="_blank"
        >
          WhatsApp
        </a>
      )}
    </div>
  );
}