import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

const FILE = path.join(process.cwd(), "content", "newsletter.json");

export async function POST(req: Request) {
  const contentType = req.headers.get("content-type") || "";
  let email = "";
  if (contentType.includes("application/json")) {
    const body = (await req.json().catch(() => ({}))) as { email?: string };
    email = (body?.email || "").toString().trim().toLowerCase();
  } else if (contentType.includes("application/x-www-form-urlencoded")) {
    const form = await req.formData();
    email = (form.get("email") || "").toString().trim().toLowerCase();
  }

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ error: "Invalid email" }, { status: 400 });
  }

  try {
    const raw = await fs.readFile(FILE, "utf8").catch(() => "[]");
    const arr = (JSON.parse(raw) as Array<{ email: string; createdAt: string }>);
    if (!arr.find((e) => e.email === email)) {
      arr.push({ email, createdAt: new Date().toISOString() });
      await fs.mkdir(path.dirname(FILE), { recursive: true });
      await fs.writeFile(FILE, JSON.stringify(arr, null, 2), "utf8");
    }
    return NextResponse.json({ success: true });
  } catch {
    return NextResponse.json({ error: "Failed to store" }, { status: 500 });
  }
}