import Link from "next/link";
import { getSiteConfig } from "@/lib/content";

export async function Footer() {
  const site = await getSiteConfig();
  const year = new Date().getFullYear();

  return (
    <footer className="mt-16 border-t border-black/5 bg-background">
      <div className="max-w-6xl mx-auto px-4 py-10 grid grid-cols-1 md:grid-cols-3 gap-8">
        <div>
          <div className="text-lg font-semibold">{site.brandName || "Lancers Tech"}</div>
          <div className="text-sm opacity-70">{site?.description || ""}</div>
        </div>
        <div>
          <div className="font-medium mb-2">Quick Links</div>
          <ul className="space-y-1 text-sm">
            <li><Link href="/services">Services</Link></li>
            <li><Link href="/portfolio">Portfolio</Link></li>
            <li><Link href="/blog">Blog</Link></li>
            <li><Link href="/faq">FAQ</Link></li>
            <li><Link href="/privacy">Privacy</Link></li>
            <li><Link href="/terms">Terms</Link></li>
          </ul>
        </div>
        <div>
          <div className="font-medium mb-2">Contact</div>
          <div className="text-sm">
            <div>{(site?.emails || [])[0]}</div>
            <div>{(site?.phones || [])[0]}</div>
          </div>
        </div>
      </div>
      <div className="border-t border-black/5 py-4 text-center text-xs opacity-70">
        © {year} {site.brandName || "Lancers Tech"}. All rights reserved.
      </div>
    </footer>
  );
}