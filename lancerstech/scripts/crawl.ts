/*
  Basic crawler for lancerstech.com
  - Respects robots.txt
  - Max depth: 3, Max pages: 100
  - Extracts titles, descriptions, headings, paragraphs, links, FAQs
  - Maps service pages to MDX in content/services
  - Maps blog posts to MDX in content/blog
  - Collects business info into content/site.json
  - Downloads images to public/images
*/

import axios from "axios";
import * as cheerio from "cheerio";
import robotsParser from "robots-parser";
import path from "path";
import fs from "fs/promises";
import crypto from "crypto";

const BASE = "https://lancerstech.com";
const MAX_PAGES = 100;
const MAX_DEPTH = 3;

const OUT_DIR = path.join(process.cwd(), "content");
const IMG_DIR = path.join(process.cwd(), "public", "images");

function toSlug(urlPath: string) {
  return urlPath
    .replace(/^\//, "")
    .replace(/\/$/, "")
    .split("/")
    .filter(Boolean)
    .pop() || "index";
}

async function ensureDir(p: string) {
  await fs.mkdir(p, { recursive: true });
}

async function downloadImage(url: string) {
  try {
    const res = await axios.get(url, { responseType: "arraybuffer", timeout: 20000 });
    const ext = (url.split(".").pop() || "jpg").split(/[?#]/)[0];
    const hash = crypto.createHash("md5").update(url).digest("hex").slice(0, 10);
    const filename = `${hash}.${ext}`.toLowerCase();
    await ensureDir(IMG_DIR);
    const filePath = path.join(IMG_DIR, filename);
    await fs.writeFile(filePath, Buffer.from(res.data));
    return `/images/${filename}`;
  } catch {
    return null;
  }
}

async function writeJson(file: string, data: any) {
  await ensureDir(path.dirname(file));
  await fs.writeFile(file, JSON.stringify(data, null, 2), "utf8");
}

async function writeText(file: string, data: string) {
  await ensureDir(path.dirname(file));
  await fs.writeFile(file, data, "utf8");
}

function normalizeUrl(href: string, base: string) {
  try {
    return new URL(href, base).toString();
  } catch {
    return null;
  }
}

function isInternal(url: string) {
  try {
    const u = new URL(url);
    const b = new URL(BASE);
    return u.hostname === b.hostname;
  } catch {
    return false;
  }
}

async function fetchRobots() {
  try {
    const { data } = await axios.get(`${BASE}/robots.txt`, { timeout: 10000 });
    return robotsParser(`${BASE}/robots.txt`, data);
  } catch {
    return robotsParser(`${BASE}/robots.txt`, "");
  }
}

function isLikelyServicePath(p: string): boolean {
  if (/^\/(about|contact|privacy|privacy-policy|terms|faq|blog)(\/|$)/i.test(p)) return false;
  if (/^\/services(\-[^/]+)?\//i.test(p)) return true;
  if (/^\/(web-development|digital-marketing|content-writing|graphic-designing|customer-services|app-design|seo)\/?$/i.test(p)) return true;
  return false;
}

async function crawl() {
  const robots = await fetchRobots();
  const queue: { url: string; depth: number }[] = [{ url: BASE, depth: 0 }];
  const visited = new Set<string>();

  const siteInfo: any = { brandName: "Lancers Tech", emails: [], phones: [], socials: {}, navigation: [] };
  const services: string[] = [];
  const blogs: string[] = [];
  const faqs: { question: string; answer: string }[] = [];

  while (queue.length && visited.size < MAX_PAGES) {
    const { url, depth } = queue.shift()!;
    if (visited.has(url)) continue;
    visited.add(url);

    const pathUrl = new URL(url).pathname;
    if (!robots.isAllowed(url, "*") || depth > MAX_DEPTH) continue;

    console.log(`Fetching: ${url}`);
    let html = "";
    try {
      const res = await axios.get(url, { timeout: 20000 });
      html = res.data as string;
    } catch {
      continue;
    }

    const $ = cheerio.load(html);

    // Basic meta
    const title = $("title").first().text().trim();
    const metaDescription = $('meta[name="description"]').attr("content") || "";

    // Collect site info on home/footer
    if (pathUrl === "/" || $("footer").length) {
      const footer = $("footer");
      const text = footer.text();
      const emailMatches = [...new Set((text.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi) || []))];
      const phoneMatches = [...new Set((text.match(/\+?\d[\d\s().-]{7,}/g) || []))];
      if (emailMatches.length) siteInfo.emails = Array.from(new Set([...(siteInfo.emails || []), ...emailMatches]));
      if (phoneMatches.length) siteInfo.phones = Array.from(new Set([...(siteInfo.phones || []), ...phoneMatches]));

      // Socials
      footer.find('a[href]').each((_, el) => {
        const href = $(el).attr('href') || '';
        if (/facebook\.com/i.test(href)) siteInfo.socials.facebook = href;
        if (/instagram\.com/i.test(href)) siteInfo.socials.instagram = href;
        if (/linkedin\.com/i.test(href)) siteInfo.socials.linkedin = href;
        if (/youtube\.com|youtu\.be/i.test(href)) siteInfo.socials.youtube = href;
        if (/tiktok\.com/i.test(href)) siteInfo.socials.tiktok = href;
        if (/wa\.me\//i.test(href)) siteInfo.whatsappNumber = href.split("wa.me/")[1]?.replace(/\D/g, "");
      });
    }

    if (pathUrl === "/") {
      siteInfo.tagline = siteInfo.tagline || $("h1").first().text().trim() || title;
      siteInfo.description = siteInfo.description || metaDescription;
    }

    // Extract images
    const imgMap = new Map<string, string>();
    await Promise.all($("img").map((_, el) => {
      const src = $(el).attr("src") || "";
      const abs = normalizeUrl(src, url);
      if (!abs) return Promise.resolve();
      return downloadImage(abs).then((local) => { if (local) imgMap.set(abs, local); });
    }).get());

    // Extract text content for MDX
    const parts: string[] = [];
    $("h1, h2, h3, p, li").each((_, el) => {
      const tag = el.tagName.toLowerCase();
      const text = $(el).text().trim();
      if (!text) return;
      if (tag === "h1") parts.push(`# ${text}`);
      else if (tag === "h2") parts.push(`## ${text}`);
      else if (tag === "h3") parts.push(`### ${text}`);
      else if (tag === "li") parts.push(`- ${text}`);
      else parts.push(text);
    });

    // FAQs (basic)
    $("details").each((_, el) => {
      const q = $(el).find("summary").text().trim();
      const a = $(el).text().replace(q, "").trim();
      if (q && a) faqs.push({ question: q, answer: a });
    });

    const body = parts.join("\n\n");

    // Services
    if (isLikelyServicePath(pathUrl)) {
      const slug = toSlug(pathUrl);
      const fm = [
        `title: ${JSON.stringify($("h1").first().text().trim() || title)}`,
        `description: ${JSON.stringify(metaDescription || "")}`,
        `slug: ${slug}`,
        `icon: ''`,
        `ctaLabel: 'Get a Quote'`,
      ].join("\n");
      const mdx = `---\n${fm}\n---\n\n${body}\n`;
      await writeText(path.join(OUT_DIR, "services", `${slug}.mdx`), mdx);
      services.push(slug);
    }

    // Blog placeholder (none detected on source; keep for future)
    if (/\/blog\//.test(pathUrl)) {
      const slug = toSlug(pathUrl);
      const date = $('meta[property="article:published_time"]').attr("content") || "";
      const fm = [
        `title: ${JSON.stringify($("h1").first().text().trim() || title)}`,
        `description: ${JSON.stringify(metaDescription || "")}`,
        `slug: ${slug}`,
        `date: ${JSON.stringify(date)}`,
      ].join("\n");
      const mdx = `---\n${fm}\n---\n\n${body}\n`;
      await writeText(path.join(OUT_DIR, "blog", `${slug}.mdx`), mdx);
      blogs.push(slug);
    }

    // Enqueue internal links
    $("a[href]").each((_, el) => {
      const href = $(el).attr("href") || "";
      const abs = normalizeUrl(href, url);
      if (!abs || !isInternal(abs)) return;
      if (!visited.has(abs)) queue.push({ url: abs, depth: depth + 1 });
    });
  }

  // Persist site.json
  const sitePath = path.join(OUT_DIR, "site.json");
  const existingSite = await fs.readFile(sitePath, "utf8").catch(() => null);
  let merged: any = existingSite ? JSON.parse(existingSite) : {};
  merged = {
    ...merged,
    ...siteInfo,
  };
  await writeJson(sitePath, merged);

  // Persist FAQs
  if (faqs.length) await writeJson(path.join(OUT_DIR, "faq.json"), faqs);

  console.log(`Crawl complete. Services: ${services.length}, Blog posts: ${blogs.length}, FAQs: ${faqs.length}`);
}

crawl().catch((e) => {
  console.error(e);
  process.exit(1);
});