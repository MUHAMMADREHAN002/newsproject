import Link from "next/link";

export function ServiceCard({ title, description, href, icon }: { title: string; description?: string; href: string; icon?: string }) {
  return (
    <Link href={href} className="block border border-black/10 rounded-lg p-5 hover:shadow-sm transition bg-white">
      <div className="text-sm opacity-60">{icon}</div>
      <div className="font-medium">{title}</div>
      {description && <p className="text-sm mt-1 opacity-80 line-clamp-3">{description}</p>}
      <div className="mt-3 text-sm text-blue-600">Learn more →</div>
    </Link>
  );
}