"use client";

import { useForm } from "react-hook-form";
import { useEffect, useState } from "react";

type FormValues = { name: string; email: string; phone?: string; message: string };

declare global {
  interface Window {
    grecaptcha?: { execute: (key: string, opts: { action: string }) => Promise<string> };
  }
}

export default function ContactForm() {
  const { register, handleSubmit, reset } = useForm<FormValues>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    const siteKey = process.env.NEXT_PUBLIC_RECAPTCHA_SITE_KEY;
    if (!siteKey) return;
    const script = document.createElement("script");
    script.src = `https://www.google.com/recaptcha/api.js?render=${siteKey}`;
    script.async = true;
    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  const onSubmit = async (data: FormValues) => {
    setSubmitting(true);
    setError(null);
    setSuccess(null);

    let token = "";
    const siteKey = process.env.NEXT_PUBLIC_RECAPTCHA_SITE_KEY;
    if (siteKey && window.grecaptcha) {
      try {
        token = await window.grecaptcha.execute(siteKey, { action: "submit" });
      } catch {}
    }

    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...data, token }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error((json && json.error) || "Failed to submit");
      setSuccess("Thanks! We will be in touch.");
      reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 max-w-xl">
      <div>
        <label className="block text-sm mb-1">Name</label>
        <input className="w-full border rounded px-3 py-2" {...register("name", { required: true })} />
      </div>
      <div>
        <label className="block text-sm mb-1">Email</label>
        <input type="email" className="w-full border rounded px-3 py-2" {...register("email", { required: true })} />
      </div>
      <div>
        <label className="block text-sm mb-1">Phone</label>
        <input className="w-full border rounded px-3 py-2" {...register("phone") } />
      </div>
      <div>
        <label className="block text-sm mb-1">Message</label>
        <textarea className="w-full border rounded px-3 py-2" rows={5} {...register("message", { required: true })} />
      </div>
      <button disabled={submitting} className="bg-black text-white px-4 py-2 rounded disabled:opacity-50">
        {submitting ? "Sending..." : "Send Message"}
      </button>
      {error && <div className="text-red-600 text-sm">{error}</div>}
      {success && <div className="text-green-600 text-sm">{success}</div>}
    </form>
  );
}