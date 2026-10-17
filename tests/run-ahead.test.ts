import { describe, expect, it } from "vitest";
import { runAhead } from "../src/gpu/run-ahead";

/** A fake GPU queue: each done() promise resolves only when the test says the GPU finished it. */
function fakeQueue() {
  const finish: (() => void)[] = [];
  const done = () => new Promise<void>((resolve) => finish.push(resolve));
  return { done, finish };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("runAhead", () => {
  it("encodes the next piece while the GPU still runs the previous one, but no further", async () => {
    const queue = fakeQueue();
    const log: string[] = [];
    const run = runAhead(3, (i) => log.push(`encode ${i}`), queue.done, (i) => log.push(`done ${i}`));
    await tick();
    expect(log).toEqual(["encode 0", "encode 1"]); // 1 runs ahead; 2 waits for 0

    queue.finish[0]();
    await tick();
    expect(log).toEqual(["encode 0", "encode 1", "done 0", "encode 2"]);

    queue.finish[1]();
    queue.finish[2]();
    await run;
    expect(log.slice(-2)).toEqual(["done 1", "done 2"]);
  });

  it("waits for every piece before returning", async () => {
    const queue = fakeQueue();
    let finished = false;
    const run = runAhead(1, () => {}, queue.done).then(() => (finished = true));
    await tick();
    expect(finished).toBe(false);
    queue.finish[0]();
    await run;
    expect(finished).toBe(true);
  });

  it("with depth 0 waits for each piece before encoding the next (the old behaviour)", async () => {
    const queue = fakeQueue();
    const log: string[] = [];
    const run = runAhead(2, (i) => log.push(`encode ${i}`), queue.done, undefined, 0);
    await tick();
    expect(log).toEqual(["encode 0"]);
    queue.finish[0]();
    await tick();
    expect(log).toEqual(["encode 0", "encode 1"]);
    queue.finish[1]();
    await run;
  });
});
