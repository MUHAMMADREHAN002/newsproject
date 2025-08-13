import Link from "next/link";

export function CTA({ title, subtitle, ctaLabel = "Contact Us", ctaHref = "/contact" }: { title: string; subtitle?: string; ctaLabel?: string; ctaHref?: string }) {
  return (
    <section className="bg-black text-white">
      <div className="max-w-6xl mx-auto px-4 py-10 md:py-14">
        <h3 className="text-2xl md:text-3xl font-semibold">{title}</h3>
        {subtitle && <p className="opacity-80 mt-2">{subtitle}</p>}
        <Link href={ctaHref} className="inline-block mt-6 bg-white text-black px-4 py-2 rounded-md">
          {ctaLabel}
        </Link>
      </div>
    </section>
  );
}