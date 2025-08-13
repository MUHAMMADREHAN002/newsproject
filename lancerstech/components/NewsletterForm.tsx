export default function NewsletterForm() {
  return (
    <form action="/api/newsletter" method="post" className="flex gap-2">
      <input
        type="email"
        name="email"
        required
        placeholder="Your email"
        className="border rounded px-3 py-2 min-w-0 flex-1"
      />
      <button className="bg-black text-white px-4 py-2 rounded">Subscribe</button>
    </form>
  );
}