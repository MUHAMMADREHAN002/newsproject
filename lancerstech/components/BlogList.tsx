import { getAllBlogPosts } from "@/lib/content";
import { BlogCard } from "./BlogCard";

export async function BlogList() {
  const posts = await getAllBlogPosts();
  if (posts.length === 0) return <div className="opacity-70">No posts yet.</div>;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {posts.map((p) => (
        <BlogCard
          key={p.frontmatter.slug}
          title={p.frontmatter.title}
          href={`/blog/${p.frontmatter.slug}`}
          description={p.frontmatter.description}
          date={p.frontmatter.date}
        />
      ))}
    </div>
  );
}