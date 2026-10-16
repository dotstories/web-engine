/**
 * Krea 2 Turbo sampling: flow matching, Euler steps, 8 steps, no guidance.
 * Sigmas go from 1 (noise) to 1/steps, with the fixed time shift mu = 1.15 of the distilled model,
 * then 0 (diffusers Krea2: linspace(1, 1/steps, steps), FlowMatchEulerDiscreteScheduler, mu = 1.15).
 */
export const TURBO = { steps: 8, mu: 1.15 };

/** Exponential time shift: σ' = e^mu / (e^mu + 1/σ − 1). */
export function sigmas(steps = TURBO.steps, mu = TURBO.mu): number[] {
  const shift = (s: number) => Math.exp(mu) / (Math.exp(mu) + (1 / s - 1));
  const raw = Array.from({ length: steps }, (_, i) => 1 - (i * (1 - 1 / steps)) / (steps - 1 || 1));
  return [...raw.map(shift), 0];
}

/** One Euler step of the flow ODE: x + (σ_next − σ) · v. */
export function eulerStep(x: Float32Array, velocity: Float32Array, sigma: number, next: number): Float32Array<ArrayBuffer> {
  return Float32Array.from(x, (v, i) => v + (next - sigma) * velocity[i]);
}

/** Seeded standard normal noise (Box–Muller), for repeatable runs. */
export function gaussian(n: number, seed: number): Float32Array<ArrayBuffer> {
  let s = seed >>> 0 || 1;
  const uniform = () => ((s = (s * 1664525 + 1013904223) >>> 0) + 0.5) / 2 ** 32;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const [r, a] = [Math.sqrt(-2 * Math.log(uniform())), 2 * Math.PI * uniform()];
    out[i] = r * Math.cos(a);
    if (i + 1 < n) out[i + 1] = r * Math.sin(a);
  }
  return out;
}
