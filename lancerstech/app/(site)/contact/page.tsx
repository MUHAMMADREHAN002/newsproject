import ContactForm from "@/components/ContactForm";
import { Section } from "@/components/Section";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export const metadata: Metadata = buildMetadata({ title: "Contact - Lancers Tech" });

export default function ContactPage() {
  return (
    <Section title="Contact Us" subtitle="We usually reply within 1 business day.">
      <ContactForm />
    </Section>
  );
}