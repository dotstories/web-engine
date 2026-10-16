/** How close a result is to a reference: cosine similarity, largest error, and error relative to the reference's size. */
export function compare(result: ArrayLike<number>, reference: ArrayLike<number>) {
  if (result.length !== reference.length) throw new Error(`Lengths differ: ${result.length} vs ${reference.length}`);
  let dot = 0;
  let a2 = 0;
  let b2 = 0;
  let maxError = 0;
  for (let i = 0; i < result.length; i++) {
    const [a, b] = [result[i], reference[i]];
    dot += a * b;
    a2 += a * a;
    b2 += b * b;
    maxError = Math.max(maxError, Math.abs(a - b));
  }
  const err2 = Array.from({ length: result.length }, (_, i) => (result[i] - reference[i]) ** 2).reduce((s, v) => s + v, 0);
  return { cosine: dot / Math.sqrt(a2 * b2), maxError, relativeError: Math.sqrt(err2 / b2) };
}
