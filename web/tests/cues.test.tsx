import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BidiText, readingSpeed, sourceDir, targetDir, visibleCharacters } from "@/lib/cues";
import { parseTimecode, safeNext, timecode } from "@/lib/utils";

describe("subtitle timing and direction", () => {
  it("round-trips timestamps and rejects invalid clock fields", () => {
    expect(timecode(3_726_045)).toBe("01:02:06.045");
    expect(parseTimecode("01:02:06.045")).toBe(3_726_045);
    expect(parseTimecode("00:00:06,045")).toBe(6045);
    expect(parseTimecode("00:60:06.045")).toBeNull();
    expect(parseTimecode("00:00:61.045")).toBeNull();
    expect(parseTimecode("not a time")).toBeNull();
  });
  it("counts graphemes rather than controls, marks, or code points", () => {
    expect(visibleCharacters("سڵاو \u2066Mira\u2069\n" )).toBe(9);
    expect(visibleCharacters("e\u0301👍🏽")).toBe(2);
    expect(readingSpeed("e\u0301👍🏽", 1000, 2000)).toBe(2);
    expect(readingSpeed("anything", 1000, 1000)).toBe(0);
  });
  it("isolates Latin runs in displayed Sorani and uses the right direction", () => {
    const { container } = render(<BidiText text="سڵاو Mira 2026" direction="rtl" />);
    expect(container.querySelector("span")).toHaveAttribute("dir", "rtl");
    expect(screen.getByText("Mira").tagName).toBe("BDI");
    expect(screen.getByText("2026")).toHaveAttribute("dir", "ltr");
    expect(sourceDir("ckb-en")).toBe("rtl");
    expect(targetDir("ckb-en")).toBe("ltr");
    expect(targetDir("en-ckb")).toBe("rtl");
  });
  it("does not allow external login redirects", () => {
    expect(safeNext("/projects/123?tab=qa")).toBe("/projects/123?tab=qa");
    expect(safeNext("//evil.example/path")).toBe("/dashboard");
    expect(safeNext("/\\evil.example/path")).toBe("/dashboard");
    expect(safeNext("https://evil.example")).toBe("/dashboard");
    expect(safeNext(null)).toBe("/dashboard");
  });
});
