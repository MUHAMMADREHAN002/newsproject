import { getBlogBySlug, listMdxFilesIn, renderMarkdownToHtml } from "@/lib/content";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export async function generateStaticParams() {
  const files = await listMdxFilesIn("blog");
  return files.map((f) => ({ slug: f.replace(/\.mdx$/, "") }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const data = await getBlogBySlug(slug);
  if (!data) return buildMetadata({ title: "Blog - Lancers Tech" });
  return buildMetadata({
    title: data.frontmatter.seo?.title || `${data.frontmatter.title} - Lancers Tech` ,
    description: data.frontmatter.seo?.description || data.frontmatter.description,
    type: "article",
  });
}

export default async function BlogPost({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const data = await getBlogBySlug(slug);
  if (!data) return notFound();
  const html = await renderMarkdownToHtml(data.content);
  return (
    <article className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-semibold">{data.frontmatter.title}</h1>
      {data.frontmatter.date && <div className="text-xs opacity-60 mt-1">{data.frontmatter.date}</div>}
      {data.frontmatter.description && <p className="mt-2 opacity-80">{data.frontmatter.description}</p>}
      <div className="mt-8 prose max-w-none" dangerouslySetInnerHTML={{ __html: html }} />
    </article>
  );
}