/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  experimental: {
    // the on-disk dev cache hit an internal Turbopack panic here after
    // files changed between runs; the app is small enough to compile cold
    turbopackFileSystemCacheForDev: false,
  },
};
