import type { NextConfig } from "next";

// The browser always calls same-origin /api URLs. The Next server alone knows
// the internal API address, so previews and containers never call localhost
// from a user's browser.
const internalApi = process.env.INTERNAL_API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["*.e2b.app", "localhost", "127.0.0.1"],
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${internalApi}/api/:path*` }];
  },
};

export default nextConfig;
