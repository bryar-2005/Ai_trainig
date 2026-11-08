"use client";
import { create } from "zustand";

// Only an in-memory handoff between the dashboard dropzone and the wizard.
// Files are never persisted to localStorage or sent until the authenticated API call.
export const useUploadStore = create<{ file: File | null; setFile: (file: File | null) => void }>((set) => ({ file: null, setFile: (file) => set({ file }) }));
