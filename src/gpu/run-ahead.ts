/**
 * Encodes `count` pieces of GPU work (e.g. transformer blocks) while the GPU runs the previous one, so it never
 * waits for the CPU. At most `depth` pieces are queued beyond the one running, which keeps memory bounded.
 * `encode(i)` must queue and submit piece i; `done()` resolves when everything submitted so far has finished
 * (device.queue.onSubmittedWorkDone). `onDone(i)` fires once piece i really finished on the GPU.
 */
export async function runAhead(
  count: number,
  encode: (i: number) => void,
  done: () => Promise<unknown>,
  onDone: (i: number) => void = () => {},
  depth = 1,
): Promise<void> {
  const queued: Promise<void>[] = [];
  for (let i = 0; i < count; i++) {
    encode(i);
    queued.push(done().then(() => onDone(i)));
    if (queued.length > depth) await queued.shift();
  }
  await Promise.all(queued);
}
