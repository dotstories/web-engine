import { describe, expect, it } from "vitest";
import {
  downloadFile,
  downloadModels,
  downloadsFor,
  KREA2_DOWNLOADS,
  type Download,
  type WritableFile,
  type WritableFolder,
} from "../src/pipeline/download";

/** An in-memory folder tree that behaves like the File System Access API (writes land on close()). */
function fakeFolder(files = new Map<string, Uint8Array>(), path = ""): WritableFolder & { files: Map<string, Uint8Array> } {
  const handle = (key: string): WritableFile => ({
    async getFile() {
      return new File([(files.get(key) ?? new Uint8Array()) as Uint8Array<ArrayBuffer>], key);
    },
    async createWritable({ keepExistingData = false } = {}) {
      let data = keepExistingData ? (files.get(key) ?? new Uint8Array()).slice() : new Uint8Array();
      let at = 0;
      return {
        async seek(p: number) {
          at = p;
        },
        async truncate(size: number) {
          const next = new Uint8Array(size);
          next.set(data.subarray(0, size));
          data = next;
        },
        async write(chunk: Uint8Array) {
          const next = new Uint8Array(Math.max(data.length, at + chunk.length));
          next.set(data);
          next.set(chunk, at);
          data = next;
          at += chunk.length;
        },
        async close() {
          files.set(key, data);
        },
      };
    },
    async move(name: string) {
      const target = key.slice(0, key.lastIndexOf("/") + 1) + name;
      files.set(target, files.get(key)!);
      files.delete(key);
    },
  });
  return {
    files,
    async getDirectoryHandle(name) {
      return fakeFolder(files, `${path}${name}/`);
    },
    async getFileHandle(name, { create = false } = {}) {
      const key = path + name;
      if (!files.has(key)) {
        if (!create) throw new DOMException("not found", "NotFoundError");
        files.set(key, new Uint8Array());
      }
      return handle(key);
    },
  };
}

const bytes = (n: number) => Uint8Array.from({ length: n }, (_, i) => (i * 7) % 256);

/** A fake server for one file, sent in chunks; honours Range unless told not to. */
function fakeServer(content: Uint8Array, { range = true, failAfter }: { range?: boolean; failAfter?: number } = {}) {
  const requests: { url: string; range?: string }[] = [];
  const fetch = (async (url: string, init?: RequestInit) => {
    const header = (init?.headers as Record<string, string> | undefined)?.Range;
    requests.push({ url, range: header });
    const start = range && header ? Number(/bytes=(\d+)-/.exec(header)![1]) : 0;
    const body = content.subarray(start);
    let sent = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (failAfter !== undefined && sent >= failAfter) return controller.error(new TypeError("network error"));
        if (sent >= body.length) return controller.close();
        const chunk = body.slice(sent, sent + 10);
        sent += chunk.length;
        controller.enqueue(chunk);
      },
    });
    return new Response(stream, { status: start ? 206 : 200 });
  }) as typeof globalThis.fetch;
  return { fetch, requests };
}

const file: Download = { folder: "vae", name: "model.safetensors", size: 35 };

describe("downloadFile", () => {
  it("downloads into folder/name in the ComfyUI layout and reports progress", async () => {
    const models = fakeFolder();
    const server = fakeServer(bytes(35));
    const seen: number[] = [];
    await downloadFile(models, file, (done) => seen.push(done), { fetch: server.fetch, source: "https://hf.test/repo" });
    expect(models.files.get("vae/model.safetensors")).toEqual(bytes(35));
    expect(models.files.has("vae/model.safetensors.part")).toBe(false);
    expect(server.requests).toEqual([{ url: "https://hf.test/repo/vae/model.safetensors", range: undefined }]);
    expect(seen).toEqual([0, 10, 20, 30, 35]);
  });

  it("keeps what arrived when the connection drops, and continues from there next time", async () => {
    const models = fakeFolder();
    await expect(downloadFile(models, file, () => {}, { fetch: fakeServer(bytes(35), { failAfter: 20 }).fetch })).rejects.toThrow("network error");
    expect(models.files.get("vae/model.safetensors.part")).toEqual(bytes(20));
    expect(models.files.has("vae/model.safetensors")).toBe(false); // a half file is never used

    const server = fakeServer(bytes(35));
    await downloadFile(models, file, () => {}, { fetch: server.fetch });
    expect(server.requests[0].range).toBe("bytes=20-");
    expect(models.files.get("vae/model.safetensors")).toEqual(bytes(35));
  });

  it("starts over when the server ignores the Range request", async () => {
    const models = fakeFolder(new Map([["vae/model.safetensors.part", bytes(20)]]));
    await downloadFile(models, file, () => {}, { fetch: fakeServer(bytes(35), { range: false }).fetch });
    expect(models.files.get("vae/model.safetensors")).toEqual(bytes(35));
  });

  it("refuses a file of the wrong size and leaves it as .part", async () => {
    const models = fakeFolder();
    await expect(downloadFile(models, file, () => {}, { fetch: fakeServer(bytes(30)).fetch })).rejects.toThrow("incomplete (30 of 35 bytes)");
    expect(models.files.has("vae/model.safetensors")).toBe(false);
  });

  it("stops on cancel and keeps the part for later", async () => {
    const models = fakeFolder();
    const controller = new AbortController();
    const server = fakeServer(bytes(35));
    const fetch = ((url: string, init?: RequestInit) => {
      expect(init?.signal).toBe(controller.signal);
      controller.abort();
      return server.fetch(url, init).then(() => Promise.reject(new DOMException("aborted", "AbortError")));
    }) as typeof globalThis.fetch;
    await expect(downloadFile(models, file, () => {}, { fetch, signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(models.files.has("vae/model.safetensors.part")).toBe(true);
  });
});

describe("downloadModels", () => {
  it("downloads the missing files one by one with one overall progress", async () => {
    const models = fakeFolder();
    const a: Download = { folder: "vae", name: "a", size: 15 };
    const b: Download = { folder: "loras", name: "b", size: 15 };
    const last: string[] = [];
    await downloadModels(models, [a, b], (done, total, name) => last.push(`${name} ${done}/${total}`), { fetch: fakeServer(bytes(15)).fetch });
    expect(models.files.get("vae/a")).toEqual(bytes(15));
    expect(models.files.get("loras/b")).toEqual(bytes(15));
    expect(last.at(-1)).toBe("b 30/30");
  });
});

describe("downloadsFor", () => {
  it("maps findModelFiles' missing list to downloads with exact sizes", () => {
    expect(downloadsFor(["vae/qwen_image_vae.safetensors"])).toEqual([KREA2_DOWNLOADS.vae]);
    expect(KREA2_DOWNLOADS.transformer).toEqual({ folder: "diffusion_models", name: "krea2_turbo_fp8_scaled.safetensors", size: 13141730784 });
  });
});
