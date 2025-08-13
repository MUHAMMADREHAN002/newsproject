/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "lancerstech.com" },
      { protocol: "https", hostname: "www.lancerstech.com" },
    ],
  },
};

module.exports = nextConfig;
