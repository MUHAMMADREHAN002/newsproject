import Link from "next/link";
import { getSiteConfig } from "@/lib/content";

export async function Header() {
  const site = await getSiteConfig();
  const nav = site.navigation || [
    { label: "Home", href: "/" },
    { label: "Services", href: "/services" },
    { label: "Portfolio", href: "/portfolio" },
    { label: "Blog", href: "/blog" },
    { label: "About", href: "/about" },
    { label: "Contact", href: "/contact" },
  ];

  return (
    <header className="sticky top-0 z-50 bg-background/80 backdrop-blur border-b border-black/5">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link href="/" className="font-semibold text-lg">
          {site.brandName || "Lancers Tech"}
        </Link>
        <nav className="hidden md:flex gap-6" aria-label="Main Navigation">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className="text-sm hover:opacity-80 transition">
              {item.label}
            </Link>
          ))}
        </nav>
        <details className="md:hidden">
          <summary aria-label="Open menu" className="cursor-pointer select-none">Menu</summary>
          <nav className="mt-2 border-t border-black/5 bg-background flex flex-col gap-2 py-2" aria-label="Mobile Navigation">
            {nav.map((item) => (
              <Link key={item.href} href={item.href} className="px-1 py-2">
                {item.label}
              </Link>
            ))}
          </nav>
        </details>
      </div>
    </header>
  );
}