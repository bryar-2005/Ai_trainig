import { afterEach, describe, expect, it } from "vitest";
import { clearDrafts, forgetDraft, pendingDrafts, rememberDraft } from "@/lib/draft-store";

afterEach(clearDrafts);
describe("ephemeral editor drafts", () => {
  it("keeps failed drafts per project and does not erase newer text on an older save", () => {
    rememberDraft("one", "cue", "first");
    rememberDraft("two", "cue", "other");
    rememberDraft("one", "cue", "newer");
    forgetDraft("one", "cue", "first");
    expect(pendingDrafts("one")).toEqual({ cue: "newer" });
    forgetDraft("one", "cue", "newer");
    expect(pendingDrafts("one")).toEqual({});
    expect(pendingDrafts("two")).toEqual({ cue: "other" });
    clearDrafts();
    expect(pendingDrafts("two")).toEqual({});
  });
});
