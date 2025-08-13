import nodemailer from "nodemailer";
import fs from "fs/promises";
import path from "path";

export type ContactPayload = {
  name: string;
  email: string;
  phone?: string;
  message: string;
};

const SUBMISSIONS_FILE = path.join(process.cwd(), "content", "submissions.json");

async function appendSubmission(payload: ContactPayload) {
  try {
    const raw = await fs.readFile(SUBMISSIONS_FILE, "utf8");
    const arr = JSON.parse(raw) as Array<Record<string, unknown>>;
    arr.push({ ...payload, createdAt: new Date().toISOString() });
    await fs.writeFile(SUBMISSIONS_FILE, JSON.stringify(arr, null, 2), "utf8");
  } catch {
    const arr: Array<Record<string, unknown>> = [{ ...payload, createdAt: new Date().toISOString() }];
    await fs.mkdir(path.dirname(SUBMISSIONS_FILE), { recursive: true });
    await fs.writeFile(SUBMISSIONS_FILE, JSON.stringify(arr, null, 2), "utf8");
  }
}

export async function sendContactEmail(payload: ContactPayload) {
  await appendSubmission(payload);

  const host = process.env.SMTP_HOST;
  const port = Number(process.env.SMTP_PORT || 587);
  const user = process.env.SMTP_USER;
  const pass = process.env.SMTP_PASS;
  const to = process.env.MAIL_TO || "info@lancerstech.com";

  if (!host || !user || !pass) {
    console.warn("SMTP not configured. Skipping email send.");
    return { success: true, skippedEmail: true } as const;
  }

  const transporter = nodemailer.createTransport({
    host,
    port,
    secure: port === 465,
    auth: { user, pass },
  });

  const subject = `New contact form submission from ${payload.name}`;
  const text = `Name: ${payload.name}\nEmail: ${payload.email}\nPhone: ${payload.phone || "-"}\n\n${payload.message}`;

  await transporter.sendMail({
    from: user,
    to,
    subject,
    text,
  });

  return { success: true } as const;
}