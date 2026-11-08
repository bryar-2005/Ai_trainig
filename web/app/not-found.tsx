import Link from "next/link";
export default function NotFound() { return <main className="grid min-h-screen place-items-center p-8"><div className="text-center"><p className="eyebrow">404 / END OF REEL</p><h1 className="mt-4 text-3xl font-semibold">This page was not found.</h1><Link href="/dashboard" className="btn-primary mt-6">Return to workspace</Link></div></main>; }
