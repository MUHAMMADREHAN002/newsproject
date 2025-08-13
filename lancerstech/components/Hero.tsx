import Link from "next/link";

export function Hero({ title, subtitle, ctaLabel = "Get Started", ctaHref = "/contact" }: { title: string; subtitle?: string; ctaLabel?: string; ctaHref?: string }) {
  return (
    <section className="relative overflow-hidden">
      <div className="max-w-6xl mx-auto px-4 py-16 md:py-24">
        <h1 className="text-3xl md:text-5xl font-bold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-4 text-base md:text-lg opacity-80 max-w-2xl">{subtitle}</p>}
        <div className="mt-8">
          <Link href={ctaHref} className="inline-block bg-black text-white px-5 py-3 rounded-md hover:opacity-90">
            {ctaLabel}
          </Link>
        </div>
      </div>
    </section>
  );
}