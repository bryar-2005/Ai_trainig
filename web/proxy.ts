import { NextRequest, NextResponse } from "next/server";

// The browser uses this same-origin proxy, never the private API address. A
// rewritten Origin is accepted only after verifying it matches the public host;
// this keeps the API's origin check meaningful behind previews and TLS proxies.
const apiOrigin = new URL(process.env.INTERNAL_API_ORIGIN ?? "http://127.0.0.1:8000").origin;

export function proxy(request: NextRequest) {
  if (request.nextUrl.pathname.startsWith("/api/")) {
    const origin = request.headers.get("origin");
    if (!origin) return NextResponse.next();
    try {
      const source = new URL(origin);
      const host = request.headers.get("x-forwarded-host")?.split(",")[0].trim() ?? request.headers.get("host");
      const protocol = request.headers.get("x-forwarded-proto")?.split(",")[0].trim() ?? request.nextUrl.protocol.slice(0, -1);
      if (!host || source.host.toLowerCase() !== host.toLowerCase() || source.protocol !== `${protocol}:`) {
        return NextResponse.json({ detail: "Origin is not allowed" }, { status: 403 });
      }
      const headers = new Headers(request.headers);
      headers.set("origin", apiOrigin);
      // Uvicorn otherwise interprets the outer proxy's HTTPS scheme as the
      // internal API scheme and rejects its own verified origin.
      headers.set("x-forwarded-proto", new URL(apiOrigin).protocol.slice(0, -1));
      return NextResponse.next({ request: { headers } });
    } catch {
      return NextResponse.json({ detail: "Origin is not allowed" }, { status: 403 });
    }
  }
  // Cookie presence is an early navigation guard only. The backend validates
  // the signature, expiry, revocation, role and ownership on every API request.
  const hasSession = request.cookies.has("sorani_access") || request.cookies.has("sorani_refresh");
  if (!hasSession) {
    const login = new URL("/login", request.url);
    login.searchParams.set("next", request.nextUrl.pathname + request.nextUrl.search);
    return NextResponse.redirect(login);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/api/:path*", "/dashboard/:path*", "/projects/:path*", "/jobs/:path*", "/models/:path*", "/settings/:path*"],
};
