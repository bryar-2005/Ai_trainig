"use client";
import { ErrorNotice } from "@/components/ui";
export default function WorkspaceError({ reset }: { error: Error & { digest?: string }; reset: () => void }) { return <div className="mx-auto max-w-xl py-16"><ErrorNotice message="This page could not be displayed." retry={reset} /></div>; }
