# Lancers Tech – Next.js 14/15 + TypeScript + Tailwind

Production-ready mirror scaffold of https://lancerstech.com using the Next.js App Router, MDX content, API routes, SEO, PWA, and forms.

## Prerequisites
- Node 18+
- npm

## Getting Started
1. Copy `.env.example` to `.env.local` and fill values.
2. Install deps:
```bash
npm install
```
3. Run the dev server:
```bash
npm run dev
```
4. Build for production:
```bash
npm run build && npm start
```

## Content
- `content/site.json`: Business info, navigation, hero.
- `content/services/*.mdx`: One file per service.
- `content/blog/*.mdx`: Blog posts.
- `content/faq.json`: FAQs.

## APIs
- `POST /api/contact`: Validates, verifies reCAPTCHA v3 (if configured), stores submission, emails via SMTP.
- `POST /api/newsletter`: Stores email addresses to `content/newsletter.json`.

## Crawl
To populate content and images from the source site, run:
```bash
npm run crawl
```
The crawler respects robots.txt, depth 3, up to 100 pages. Images are downloaded into `public/images` and optimized.

## SEO & PWA
- Dynamic metadata per page
- Open Graph/Twitter tags
- JSON-LD helpers in `lib/seo.ts`
- `app/sitemap.ts` and `app/robots.ts`
- `public/site.webmanifest` and `public/sw.js`

## Deploy
- Vercel recommended. Ensure environment variables and `NEXT_PUBLIC_SITE_URL` are set to the live URL.
