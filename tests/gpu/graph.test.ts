import { describe, expect, it } from "vitest";
import { getDevice } from "../../src/gpu/compute";
import { Graph } from "../../src/gpu/graph";
import { addWgsl, elementwiseWorkgroups } from "../../src/kernels/elementwise";

describe("Graph", () => {
  it("binds part of a buffer (offset in bytes, multiple of 256)", async () => {
    const graph = new Graph(await getDevice());
    const data = graph.buffer(Float32Array.from({ length: 192 }, (_, i) => i)); // 3 × 64 values
    const out = graph.buffer(64 * 4);
    graph.run(addWgsl, [graph.buffer(new Uint32Array([64, 1])), Graph.slice(data, 256, 256), Graph.slice(data, 512, 256), out], elementwiseWorkgroups(64));
    const r = new Float32Array(await graph.read(out));
    expect(r[0]).toBe(64 + 128);
    expect(r[63]).toBe(127 + 191);
    graph.destroy();
  });

  it("copies between buffers in order with the dispatches", async () => {
    const graph = new Graph(await getDevice());
    const a = graph.buffer(Float32Array.from([1, 2, 3, 4]));
    const b = graph.buffer(8 * 4);
    graph.copy(a, 0, b, 16, 16);
    expect(Array.from(new Float32Array(await graph.read(b)))).toEqual([0, 0, 0, 0, 1, 2, 3, 4]);
    graph.destroy();
  });

  it("release frees a step's temporaries but keeps what is asked for", async () => {
    const graph = new Graph(await getDevice());
    const before = graph.buffer(4);
    const mark = graph.mark();
    graph.buffer(4);
    const kept = graph.buffer(Float32Array.from([7]));
    graph.release(mark, [kept]);
    expect(graph.mark()).toBe(mark + 1); // `before`, then `kept`
    expect(new Float32Array(await graph.read(kept))[0]).toBe(7);
    graph.free(before);
    graph.destroy();
  });

  it("reuses released buffers of the same size instead of creating new ones", async () => {
    const graph = new Graph(await getDevice());
    const mark = graph.mark();
    const first = graph.buffer(4096);
    graph.release(mark);
    const again = graph.buffer(4096);
    expect(again).toBe(first);
    expect(graph.stats).toMatchObject({ created: 1, reused: 1 });
    expect(graph.buffer(8192)).not.toBe(first); // other sizes get their own buffer
    graph.destroy();
  });

  it("zeroes a reused buffer, and fills a reused buffer with the new data", async () => {
    const graph = new Graph(await getDevice());
    let mark = graph.mark();
    graph.buffer(Float32Array.from({ length: 1000 }, (_, i) => i + 1));
    graph.release(mark);
    const zeroed = graph.buffer(4000);
    expect(new Float32Array(await graph.read(zeroed)).every((v) => v === 0)).toBe(true);

    mark = graph.mark();
    graph.buffer(Float32Array.from([1, 2, 3, 4]));
    graph.release(mark);
    const filled = graph.buffer(Float32Array.from([5, 6, 7, 8]));
    expect(Array.from(new Float32Array(await graph.read(filled)))).toEqual([5, 6, 7, 8]);
    expect(graph.stats.reused).toBe(2);
    graph.destroy();
  });

  it("free() also returns the buffer to the pool", async () => {
    const graph = new Graph(await getDevice());
    const a = graph.buffer(256);
    graph.free(a);
    expect(graph.buffer(256)).toBe(a);
    graph.destroy();
  });
});
