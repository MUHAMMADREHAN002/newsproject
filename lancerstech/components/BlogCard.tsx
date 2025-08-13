import Link from "next/link";

export function BlogCard({ title, description, href, date }: { title: string; description?: string; href: string; date?: string }) {
  return (
    <Link href={href} className="block p-5 border border-black/10 rounded-lg hover:shadow-sm bg-white">
      <div className="text-xs opacity-60">{date}</div>
      <div className="font-medium mt-1">{title}</div>
      {description && <p className="text-sm mt-1 opacity-80 line-clamp-3">{description}</p>}
      <div className="mt-3 text-sm text-blue-600">Read more →</div>
    </Link>
  );
}