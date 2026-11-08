import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter Variable", "sans-serif"],
        mono: ["JetBrains Mono Variable", "monospace"],
        arabic: ["Vazirmatn Variable", "Noto Naskh Arabic Variable", "sans-serif"],
      },
      colors: {
        ink: "var(--ink)", muted: "var(--muted)", surface: "var(--surface)",
        canvas: "var(--canvas)", line: "var(--line)", accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
      },
      boxShadow: { panel: "0 12px 36px -24px rgba(25, 34, 30, .24)" },
    },
  },
  plugins: [],
};
export default config;
