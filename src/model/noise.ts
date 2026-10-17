/**
 * The same random noise as ComfyUI: `torch.manual_seed(seed)` then `torch.randn(shape)` on the CPU in float32
 * (comfy/sample.py prepare_noise). Same seed → same starting noise → the same picture as ComfyUI makes.
 */

const N = 624;

/** Mersenne Twister (MT19937), the generator behind torch.manual_seed on the CPU. */
export class MT19937 {
  private readonly state = new Uint32Array(N);
  private index = N;

  /** torch keeps the low 32 bits of the seed. */
  constructor(seed: number) {
    const s = this.state;
    s[0] = seed >>> 0;
    for (let i = 1; i < N; i++) s[i] = (Math.imul(1812433253, s[i - 1] ^ (s[i - 1] >>> 30)) + i) >>> 0;
  }

  /** Next 32-bit number (torch: CPUGeneratorImpl::random). */
  next(): number {
    if (this.index >= N) this.twist();
    let y = this.state[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  private twist() {
    const s = this.state;
    for (let i = 0; i < N; i++) {
      const y = (s[i] & 0x80000000) | (s[(i + 1) % N] & 0x7fffffff);
      s[i] = s[(i + 397) % N] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
    }
    this.index = 0;
  }
}

const f32 = Math.fround;
const TWO_PI = f32(2 * Math.PI);

/** torch's uniform float in [0, 1): the low 24 bits of one draw. */
const uniform24 = (mt: MT19937) => (mt.next() & 0xffffff) * 2 ** -24;

/** torch's uniform double in [0, 1): 53 bits of two draws (random64 = first << 32 | second). */
const uniform53 = (mt: MT19937) => ((mt.next() & 0x1fffff) * 2 ** 32 + mt.next()) * 2 ** -53;

/** Box–Muller on 16 uniforms in place: pairs (j, j + 8), float math (torch normal_fill_16, AVX2 build). */
function fill16(d: Float32Array, o: number) {
  for (let j = 0; j < 8; j++) {
    const u1 = f32(1 - d[o + j]); // [0, 1) → (0, 1] for the log
    const radius = f32(Math.sqrt(f32(-2 * f32(Math.log(u1)))));
    const theta = f32(TWO_PI * d[o + j + 8]);
    d[o + j] = radius * f32(Math.cos(theta));
    d[o + j + 8] = radius * f32(Math.sin(theta));
  }
}

/** `torch.manual_seed(seed); torch.randn(n)` on the CPU, float32. */
export function torchRandn(n: number, seed: number): Float32Array<ArrayBuffer> {
  const mt = new MT19937(seed);
  const out = new Float32Array(n);
  if (n < 16) {
    // torch draws doubles one by one here; each pair gives cos, then the kept sin.
    for (let i = 0; i < n; i += 2) {
      const u1 = uniform53(mt);
      const u2 = uniform53(mt);
      const [r, theta] = [Math.sqrt(-2 * Math.log1p(-u2)), 2 * Math.PI * u1];
      out[i] = r * Math.cos(theta);
      if (i + 1 < n) out[i + 1] = r * Math.sin(theta);
    }
    return out;
  }
  for (let i = 0; i < n; i++) out[i] = uniform24(mt);
  for (let i = 0; i + 16 <= n; i += 16) fill16(out, i);
  if (n % 16) {
    // torch redoes the last 16 values with fresh draws.
    for (let i = n - 16; i < n; i++) out[i] = uniform24(mt);
    fill16(out, n - 16);
  }
  return out;
}
