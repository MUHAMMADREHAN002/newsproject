import type { Metadata } from "next";

export type SeoInput = {
  title?: string;
  description?: string;
  url?: string;
  image?: string;
  canonical?: string;
  type?: "website" | "article"; // limit to allowed Next types
};

export function buildMetadata(input: SeoInput): Metadata {
  const title = input.title || "Lancers Tech";
  const description = input.description || "";
  const url = input.url || process.env.NEXT_PUBLIC_SITE_URL || "";
  const image = input.image || "/og-image.png";
  const canonical = input.canonical || url;

  return {
    title,
    description,
    alternates: { canonical },
    openGraph: {
      title,
      description,
      url,
      type: input.type || "website",
      images: [{ url: image }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [image],
    },
    robots: {
      index: true,
      follow: true,
    },
  };
}

export function jsonLd(data: Record<string, unknown>) {
  return JSON.stringify(data);
}

export function orgJsonLd({
  name,
  url,
  logo,
  sameAs,
}: { name: string; url: string; logo?: string; sameAs?: string[] }) {
  return jsonLd({
    "@context": "https://schema.org",
    "@type": "Organization",
    name,
    url,
    logo,
    sameAs,
  });
}