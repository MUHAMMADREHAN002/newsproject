import React from "react";

export function FAQ({ items }: { items: { question: string; answer: string }[] }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="space-y-3">
      {items.map((item, idx) => (
        <details key={idx} className="border border-black/10 rounded">
          <summary className="cursor-pointer select-none p-4 font-medium">
            {item.question}
          </summary>
          <div className="p-4 pt-0 text-sm opacity-85">{item.answer}</div>
        </details>
      ))}
    </div>
  );
}