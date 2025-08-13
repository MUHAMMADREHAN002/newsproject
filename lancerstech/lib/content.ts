import fs from "fs/promises";
import path from "path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeStringify from "rehype-stringify";

export type SiteConfig = {
  brandName: string;
  tagline?: string;
  description?: string;
  emails?: string[];
  phones?: string[];
  whatsappNumber?: string;
  address?: string;
  officeHours?: string[];
  socials?: Record<string, string>;
  mapEmbed?: string;
  navigation?: { label: string; href: string }[];
  hero?: { title?: string; subtitle?: string; ctaLabel?: string; ctaHref?: string };
};

export type FAQItem = { question: string; answer: string };

export type MDXFrontmatter = {
  title: string;
  description?: string;
  slug: string;
  date?: string;
  icon?: string;
  ctaLabel?: string;
  seo?: {
    title?: string;
    description?: string;
    ogImage?: string;
  };
  schema?: Record<string, unknown>;
};

const CONTENT_DIR = path.join(process.cwd(), "content");

async function safeReadJson<T>(filePath: string, fallback: T): Promise<T> {
  try {
    const data = await fs.readFile(filePath, "utf8");
    return JSON.parse(data) as T;
  } catch {
    return fallback;
  }
}

export async function getSiteConfig(): Promise<SiteConfig> {
  const filePath = path.join(CONTENT_DIR, "site.json");
  return safeReadJson<SiteConfig>(filePath, {
    brandName: "Lancers Tech",
    tagline: "Digital Solutions",
    description: "",
    emails: ["info@lancerstech.com"],
    phones: [],
    whatsappNumber: "",
    address: "",
    officeHours: [],
    socials: {},
    navigation: [
      { label: "Home", href: "/" },
      { label: "Services", href: "/services" },
      { label: "Portfolio", href: "/portfolio" },
      { label: "Blog", href: "/blog" },
      { label: "About", href: "/about" },
      { label: "Contact", href: "/contact" },
    ],
  });
}

export async function getFAQs(): Promise<FAQItem[]> {
  const filePath = path.join(CONTENT_DIR, "faq.json");
  return safeReadJson<FAQItem[]>(filePath, []);
}

async function readMdxFile(filePath: string): Promise<{ frontmatter: MDXFrontmatter; content: string } | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    const { data, content } = matter(raw);
    const frontmatter = data as MDXFrontmatter;
    return { frontmatter, content };
  } catch {
    return null;
  }
}

export async function renderMarkdownToHtml(markdown: string): Promise<string> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype)
    .use(rehypeStringify)
    .process(markdown);
  return String(file);
}

export async function listMdxFilesIn(dir: string): Promise<string[]> {
  try {
    const fullDir = path.join(CONTENT_DIR, dir);
    const files = await fs.readdir(fullDir);
    return files.filter((f) => f.endsWith(".mdx"));
  } catch {
    return [];
  }
}

export async function getAllServices(): Promise<{ frontmatter: MDXFrontmatter }[]> {
  const files = await listMdxFilesIn("services");
  const items: { frontmatter: MDXFrontmatter }[] = [];
  for (const file of files) {
    const data = await readMdxFile(path.join(CONTENT_DIR, "services", file));
    if (data) items.push({ frontmatter: data.frontmatter });
  }
  return items.sort((a, b) => {
    const aKey = a.frontmatter.title || a.frontmatter.slug || "";
    const bKey = b.frontmatter.title || b.frontmatter.slug || "";
    return aKey.localeCompare(bKey);
  });
}

export async function getServiceBySlug(slug: string) {
  const filePath = path.join(CONTENT_DIR, "services", `${slug}.mdx`);
  return readMdxFile(filePath);
}

export async function getAllBlogPosts(): Promise<{ frontmatter: MDXFrontmatter }[]> {
  const files = await listMdxFilesIn("blog");
  const items: { frontmatter: MDXFrontmatter }[] = [];
  for (const file of files) {
    const data = await readMdxFile(path.join(CONTENT_DIR, "blog", file));
    if (data) items.push({ frontmatter: data.frontmatter });
  }
  return items.sort((a, b) => {
    const aKey = a.frontmatter.date || a.frontmatter.title || a.frontmatter.slug || "";
    const bKey = b.frontmatter.date || b.frontmatter.title || b.frontmatter.slug || "";
    return bKey.localeCompare(aKey);
  });
}

export async function getBlogBySlug(slug: string) {
  const filePath = path.join(CONTENT_DIR, "blog", `${slug}.mdx`);
  return readMdxFile(filePath);
}