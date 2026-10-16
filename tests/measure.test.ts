import { describe, expect, it } from "vitest";
import { attempt } from "../src/probe/measure";

describe("attempt", () => {
  it("returns the value when it works", async () => {
    const errors = {};
    expect(await attempt(errors, "read", async () => 42)).toBe(42);
    expect(errors).toEqual({});
  });

  it("returns null and keeps the error when it fails", async () => {
    const errors: Record<string, string> = {};
    const fail = async () => {
      throw new DOMException("denied", "NotAllowedError");
    };
    expect(await attempt(errors, "read", fail)).toBeNull();
    expect(errors).toEqual({ read: "NotAllowedError: denied" });
  });

  it("keeps errors that are not Error objects", async () => {
    const errors: Record<string, string> = {};
    await attempt(errors, "quota", () => Promise.reject("nope"));
    expect(errors).toEqual({ quota: "nope" });
  });
});
