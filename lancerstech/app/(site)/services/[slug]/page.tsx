import { getServiceBySlug, listMdxFilesIn, renderMarkdownToHtml } from "@/lib/content";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { buildMetadata } from "@/lib/seo";

export async function generateStaticParams() {
  const files = await listMdxFilesIn("services");
  return files.map((f) => ({ slug: f.replace(/\.mdx$/, "") }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const data = await getServiceBySlug(slug);
  if (!data) return buildMetadata({ title: "Service - Lancers Tech" });
  return buildMetadata({
    title: data.frontmatter.seo?.title || `${data.frontmatter.title} - Lancers Tech` ,
    description: data.frontmatter.seo?.description || data.frontmatter.description,
  });
}

export default async function ServiceDetail({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const data = await getServiceBySlug(slug);
  if (!data) return notFound();
  const html = await renderMarkdownToHtml(data.content);
  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-semibold">{data.frontmatter.title}</h1>
      {data.frontmatter.description && <p className="mt-2 opacity-80">{data.frontmatter.description}</p>}
      <div className="mt-8 prose max-w-none" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}