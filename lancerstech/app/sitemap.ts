import type { MetadataRoute } from "next";
import { getAllServices, getAllBlogPosts } from "@/lib/content";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const base = process.env.NEXT_PUBLIC_SITE_URL || "";
  const staticRoutes = ["/", "/about", "/services", "/portfolio", "/courses", "/blog", "/contact", "/faq", "/privacy", "/terms"];
  const services = await getAllServices();
  const posts = await getAllBlogPosts();

  const entries: MetadataRoute.Sitemap = [];
  for (const r of staticRoutes) entries.push({ url: `${base}${r}`, changeFrequency: "weekly", priority: 0.7 });
  for (const s of services) entries.push({ url: `${base}/services/${s.frontmatter.slug}`, changeFrequency: "monthly", priority: 0.6 });
  for (const p of posts) entries.push({ url: `${base}/blog/${p.frontmatter.slug}`, changeFrequency: "monthly", priority: 0.6 });
  return entries;
}