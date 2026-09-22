/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  // the local model service (inference/server.py); proxied so the browser
  // talks to one origin and no CORS setup is needed
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8765/api/:path*" }];
  },
};
