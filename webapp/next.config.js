/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  experimental: {
    // the on-disk dev cache hit an internal Turbopack panic here after
    // files changed between runs; the app is small enough to compile cold
    turbopackFileSystemCacheForDev: false,
  },
  // the local model service (inference/server.py); proxied so the browser
  // talks to one origin and no CORS setup is needed
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8765/api/:path*" }];
  },
};
