import { NextResponse } from "next/server";
import { getSiteConfig } from "@/lib/content";

export async function GET() {
  const site = await getSiteConfig();
  return NextResponse.json(site);
}