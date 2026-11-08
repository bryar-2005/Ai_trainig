import type { Metadata } from "next";
import { Providers } from "@/components/providers";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Sorani Studio — subtitles with context", template: "%s · Sorani Studio" },
  description: "A thoughtful workspace for English and Kurdish Sorani subtitle translation and review.",
  referrer: "no-referrer",
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" suppressHydrationWarning><body><Providers>{children}</Providers></body></html>;
}
