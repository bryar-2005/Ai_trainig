import type { User } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); this.name = "ApiError"; }
}

function csrf(): string | undefined {
  if (typeof document === "undefined") return undefined;
  const found = document.cookie.split("; ").find((part) => part.startsWith("sorani_csrf="));
  return found ? decodeURIComponent(found.slice("sorani_csrf=".length)) : undefined;
}

let refreshPromise: Promise<boolean> | null = null;
export function refreshSession(): Promise<boolean> {
  // Refresh requires the readable session-bound CSRF cookie. Avoid consuming
  // credential-attempt limits for anonymous visits with no browser session.
  if (!csrf()) return Promise.resolve(false);
  if (!refreshPromise) {
    const token = csrf();
    refreshPromise = fetch("/api/auth/refresh", {
      method: "POST", credentials: "same-origin", headers: token ? { "X-CSRF-Token": token } : {},
    }).then((response) => response.ok).catch(() => false).finally(() => { refreshPromise = null; });
  }
  return refreshPromise;
}

async function responseError(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status}).`;
  try {
    const data = await response.json();
    if (typeof data.detail === "string") message = data.detail;
    else if (typeof data.detail?.message === "string") message = data.detail.message;
    else if (Array.isArray(data.detail) && typeof data.detail[0]?.msg === "string") message = data.detail[0].msg;
  } catch { /* A non-JSON error has no additional public detail. */ }
  return new ApiError(response.status, message);
}

type Options = Omit<RequestInit, "body"> & { body?: unknown; retryAuth?: boolean };
async function request(path: string, options: Options = {}): Promise<Response> {
  const { body, retryAuth = true, ...rest } = options;
  const headers = new Headers(rest.headers);
  const method = (rest.method ?? "GET").toUpperCase();
  const token = csrf();
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && token) headers.set("X-CSRF-Token", token);
  const form = body instanceof FormData;
  if (body !== undefined && !form) headers.set("Content-Type", "application/json");
  const response = await fetch(path, {
    ...rest, method, credentials: "same-origin", headers,
    body: body === undefined ? undefined : form ? body as FormData : JSON.stringify(body),
    cache: "no-store",
  });
  if (response.status === 401 && retryAuth && !path.startsWith("/api/auth/")) {
    if (await refreshSession()) return request(path, { ...options, retryAuth: false });
    if (typeof window !== "undefined") window.dispatchEvent(new Event("session-expired"));
  }
  if (!response.ok) throw await responseError(response);
  return response;
}

export async function api<T>(path: string, options: Options = {}): Promise<T> {
  const response = await request(path, options);
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
export async function apiText(path: string, options: Options = {}): Promise<string> {
  return (await request(path, options)).text();
}
export async function download(path: string, filename: string, options: Options = {}): Promise<void> {
  const blob = await (await request(path, options)).blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function currentUser(): Promise<User | null> {
  try { return await api<User>("/api/auth/me", { retryAuth: false }); }
  catch (error) {
    if (error instanceof ApiError && error.status === 401 && await refreshSession()) {
      try { return await api<User>("/api/auth/me", { retryAuth: false }); }
      catch (reason) { if (reason instanceof ApiError && reason.status === 401) return null; throw reason; }
    }
    if (error instanceof ApiError && error.status === 401) return null;
    throw error;
  }
}
