import { NextResponse } from "next/server";
import { z } from "zod";
import { rateLimit } from "@/lib/rateLimit";
import { sendContactEmail } from "@/lib/mail";

const schema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  phone: z.string().optional(),
  message: z.string().min(5),
  token: z.string().optional(),
});

async function verifyRecaptcha(token?: string) {
  const secret = process.env.RECAPTCHA_SECRET;
  if (!secret) return true; // allow when not configured
  if (!token) return false;
  try {
    const params = new URLSearchParams();
    params.set("secret", secret);
    params.set("response", token);
    const res = await fetch("https://www.google.com/recaptcha/api/siteverify", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params,
    });
    const json = (await res.json()) as { success?: boolean } | unknown;
    if (typeof json === "object" && json && "success" in json) {
      return Boolean((json as { success?: boolean }).success);
    }
    return false;
  } catch {
    return false;
  }
}

export async function POST(req: Request) {
  const ip = (req.headers.get("x-forwarded-for") || "").split(",")[0] || "unknown";
  const rl = rateLimit(`contact:${ip}`, 5, 60_000);
  if (!rl.success) return NextResponse.json({ error: "Too many requests" }, { status: 429 });

  const body = await req.json().catch(() => null);
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: "Invalid input" }, { status: 400 });

  const ok = await verifyRecaptcha(parsed.data.token);
  if (!ok) return NextResponse.json({ error: "reCAPTCHA failed" }, { status: 400 });

  await sendContactEmail(parsed.data);
  return NextResponse.json({ success: true });
}