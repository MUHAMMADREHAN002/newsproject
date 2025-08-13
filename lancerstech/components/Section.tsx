import { ReactNode } from "react";

export function Section({ title, subtitle, children }: { title?: string; subtitle?: string; children: ReactNode }) {
  return (
    <section className="max-w-6xl mx-auto px-4 py-12 md:py-16">
      {title && <h2 className="text-2xl md:text-3xl font-semibold">{title}</h2>}
      {subtitle && <p className="opacity-80 mt-2 max-w-2xl">{subtitle}</p>}
      <div className="mt-6">{children}</div>
    </section>
  );
}