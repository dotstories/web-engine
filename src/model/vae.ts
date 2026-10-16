import type { Graph } from "../gpu/graph";
import { attention } from "../kernels/attention";
import { addBiasWgsl, elementwiseWorkgroups } from "../kernels/elementwise";
import { upsample2xWgsl, upsample2xWorkgroups } from "../kernels/image";
import { conv3x3Params, conv3x3Wgsl, matmulF32Wgsl, matmulParams, matmulWorkgroups } from "../kernels/matmul";
import { toFloat32 } from "../weights/fp8";
import { readTensor, type Safetensors } from "../weights/safetensors";
import { add, noMod, norm, silu } from "./ops";

/**
 * Krea 2's image decoder: the Qwen-Image VAE (Wan 2.1 VAE architecture, ComfyUI comfy/ldm/wan/vae.py).
 * For one image every causal 3D conv is a 2D conv with the kernel's last time slice, and the
 * temporal upsampling convs are skipped. Images are channel-last [H · W, C] throughout.
 *
 * z (16 ch) → conv2 1×1 → conv1 3×3 → 384 → middle (res, attention, res) → 4 up stages
 * (384, 384, 192, 96 channels; ×2 nearest upsample + 3×3 conv between them) → norm, SiLU, conv → RGB.
 */

/** Wan 2.1 latent format (ComfyUI latent_formats.Wan21): model latents → VAE latents = x · std + mean. */
export const LATENT_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
export const LATENT_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916];

const EPS = 1e-24; // F.normalize: x / max(‖x‖, 1e-12) · √C, i.e. a plain RMSNorm without a real eps
const ATTENTION_CHUNK = 2048;

export type Conv = { weight: GPUBuffer; bias: GPUBuffer; cin: number; cout: number; size: 1 | 3 };
export type ResBlock = { kind: "res"; norm1: GPUBuffer; conv1: Conv; norm2: GPUBuffer; conv2: Conv; shortcut?: Conv };
export type AttnBlock = { kind: "attn"; norm: GPUBuffer; q: Conv; k: Conv; v: Conv; proj: Conv };
export type Upsample = { kind: "up"; conv: Conv };
export type VaeWeights = { conv2: Conv; conv1: Conv; middle: (ResBlock | AttnBlock)[]; ups: (ResBlock | Upsample)[]; headNorm: GPUBuffer; head: Conv };

/** [C, H, W] → [H · W, C] */
export function channelLast(x: Float32Array, channels: number, pixels: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(x.length);
  for (let c = 0; c < channels; c++) for (let p = 0; p < pixels; p++) out[p * channels + c] = x[c * pixels + p];
  return out;
}

/** [H · W, C] → [C, H, W] */
export function channelFirst(x: Float32Array, channels: number, pixels: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(x.length);
  for (let c = 0; c < channels; c++) for (let p = 0; p < pixels; p++) out[c * pixels + p] = x[p * channels + c];
  return out;
}

/** The transformer's latent [16, H, W] → the decoder's input [H · W, 16] (Wan 2.1 latent format). */
export function vaeInput(latent: Float32Array, pixels: number): Float32Array<ArrayBuffer> {
  const out = channelLast(latent, 16, pixels);
  for (let i = 0; i < out.length; i++) out[i] = out[i] * LATENT_STD[i % 16] + LATENT_MEAN[i % 16];
  return out;
}

/** Decoder output [H · W, 3] in about [-1, 1] → RGBA bytes (ComfyUI: clamp((x + 1) / 2, 0, 1)). */
export function toRgba(rgb: Float32Array, pixels: number): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(pixels * 4);
  for (let p = 0; p < pixels; p++) {
    for (let c = 0; c < 3; c++) out[p * 4 + c] = Math.round(Math.min(1, Math.max(0, (rgb[p * 3 + c] + 1) / 2)) * 255);
    out[p * 4 + 3] = 255;
  }
  return out;
}

/** Conv weight [cout, cin, (t), kh, kw] → last time slice as [cout, (ky, kx, cin)] for the conv3x3 matmul. */
export function convWeight(w: Float32Array, shape: number[]): Float32Array<ArrayBuffer> {
  const [cout, cin] = shape;
  const [kt, kh, kw] = shape.length === 5 ? shape.slice(2) : [1, ...shape.slice(2)];
  const out = new Float32Array(cout * kh * kw * cin);
  for (let o = 0; o < cout; o++)
    for (let c = 0; c < cin; c++)
      for (let y = 0; y < kh; y++)
        for (let x = 0; x < kw; x++)
          out[((o * kh + y) * kw + x) * cin + c] = w[(((o * cin + c) * kt + kt - 1) * kh + y) * kw + x];
  return out;
}

async function floats(file: Blob, st: Safetensors, name: string) {
  const t = st.tensors.get(name);
  if (!t) throw new Error(`Missing tensor ${name}: is this the Qwen-Image VAE?`);
  return { data: toFloat32(t.dtype, await readTensor(file, t)), shape: t.shape };
}

export async function loadVae(graph: Graph, file: Blob, st: Safetensors): Promise<VaeWeights> {
  const conv = async (name: string, rows?: [number, number]): Promise<Conv> => {
    const w = await floats(file, st, `${name}.weight`);
    const b = await floats(file, st, `${name}.bias`);
    const [cout, cin] = w.shape;
    let weight = convWeight(w.data, w.shape);
    let bias = b.data;
    if (rows) { // part of a fused weight (to_qkv): output channels rows[0]..rows[1]
      weight = weight.slice(rows[0] * cin, rows[1] * cin);
      bias = bias.slice(rows[0], rows[1]);
    }
    return { weight: graph.buffer(weight), bias: graph.buffer(bias), cin, cout: rows ? rows[1] - rows[0] : cout, size: w.shape.at(-1) === 3 ? 3 : 1 };
  };
  const gamma = async (name: string) => graph.buffer((await floats(file, st, name)).data);
  const res = async (prefix: string): Promise<ResBlock> => ({
    kind: "res",
    norm1: await gamma(`${prefix}.residual.0.gamma`),
    conv1: await conv(`${prefix}.residual.2`),
    norm2: await gamma(`${prefix}.residual.3.gamma`),
    conv2: await conv(`${prefix}.residual.6`),
    shortcut: st.tensors.has(`${prefix}.shortcut.weight`) ? await conv(`${prefix}.shortcut`) : undefined,
  });
  const attn = async (prefix: string): Promise<AttnBlock> => {
    const c = st.tensors.get(`${prefix}.proj.weight`)!.shape[0];
    return {
      kind: "attn",
      norm: await gamma(`${prefix}.norm.gamma`),
      q: await conv(`${prefix}.to_qkv`, [0, c]),
      k: await conv(`${prefix}.to_qkv`, [c, 2 * c]),
      v: await conv(`${prefix}.to_qkv`, [2 * c, 3 * c]),
      proj: await conv(`${prefix}.proj`),
    };
  };
  const ups: (ResBlock | Upsample)[] = [];
  for (let i = 0; st.tensors.has(`decoder.upsamples.${i}.residual.0.gamma`) || st.tensors.has(`decoder.upsamples.${i}.resample.1.weight`); i++) {
    const p = `decoder.upsamples.${i}`;
    ups.push(st.tensors.has(`${p}.resample.1.weight`) ? { kind: "up", conv: await conv(`${p}.resample.1`) } : await res(p));
  }
  return {
    conv2: await conv("conv2"),
    conv1: await conv("decoder.conv1"),
    middle: [await res("decoder.middle.0"), await attn("decoder.middle.1"), await res("decoder.middle.2")],
    ups,
    headNorm: await gamma("decoder.head.0.gamma"),
    head: await conv("decoder.head.2"),
  };
}

function applyConv(graph: Graph, x: GPUBuffer, c: Conv, height: number, width: number): GPUBuffer {
  const pixels = height * width;
  const y = graph.buffer(pixels * c.cout * 4);
  if (c.size === 3) {
    const { shape, params } = conv3x3Params(height, width, c.cin, c.cout);
    graph.run(conv3x3Wgsl, [graph.buffer(params), x, c.weight, y], matmulWorkgroups(shape));
  } else {
    const shape = { m: pixels, n: c.cout, k: c.cin };
    graph.run(matmulF32Wgsl, [graph.buffer(matmulParams(shape, 1)), x, c.weight, y], matmulWorkgroups(shape));
  }
  const out = graph.buffer(pixels * c.cout * 4);
  graph.run(addBiasWgsl, [graph.buffer(new Uint32Array([pixels * c.cout, c.cout])), y, c.bias, out], elementwiseWorkgroups(pixels * c.cout));
  return out;
}

/** Per-pixel RMS norm over channels × gamma (Wan RMS_norm). */
const pixelNorm = (graph: Graph, x: GPUBuffer, gamma: GPUBuffer, pixels: number, channels: number) =>
  norm(graph, x, gamma, pixels, channels, noMod(graph, channels), EPS, 0);

function resBlock(graph: Graph, r: ResBlock, x: GPUBuffer, height: number, width: number): GPUBuffer {
  const pixels = height * width;
  const { cin, cout } = r.conv1;
  let h = applyConv(graph, silu(graph, pixelNorm(graph, x, r.norm1, pixels, cin), pixels * cin), r.conv1, height, width);
  h = applyConv(graph, silu(graph, pixelNorm(graph, h, r.norm2, pixels, cout), pixels * cout), r.conv2, height, width);
  return add(graph, h, r.shortcut ? applyConv(graph, x, r.shortcut, height, width) : x, pixels * cout);
}

function attnBlock(graph: Graph, a: AttnBlock, x: GPUBuffer, height: number, width: number): GPUBuffer {
  const pixels = height * width;
  const c = a.proj.cout;
  const n = pixelNorm(graph, x, a.norm, pixels, c);
  const [q, k, v] = [a.q, a.k, a.v].map((w) => applyConv(graph, n, w, height, width));
  const o = attention(graph, q, k, v, { tokens: pixels, heads: 1, kvHeads: 1, dim: c, chunk: ATTENTION_CHUNK });
  return add(graph, applyConv(graph, o, a.proj, height, width), x, pixels * c);
}

/**
 * Decodes z [h · w, 16] (VAE latents, see vaeInput) → RGB [8h · 8w, 3] in about [-1, 1].
 * Each stage's temporaries are freed as it goes; `keepSteps` also keeps every stage output (small sizes only).
 */
export async function decode(graph: Graph, w: VaeWeights, z: GPUBuffer, height: number, width: number, keepSteps = false, onStage?: (name: string) => void) {
  const steps: Record<string, GPUBuffer> = {};
  let x = z;
  let [h, wd] = [height, width];
  const stage = async (name: string, f: () => GPUBuffer) => {
    const mark = graph.mark();
    const next = f();
    graph.release(mark, [next]);
    if (x !== z && !keepSteps) graph.free(x);
    x = next;
    if (keepSteps) steps[name] = x;
    onStage?.(name);
    await graph.device.queue.onSubmittedWorkDone(); // bounded memory, honest progress
  };

  await stage("conv2", () => applyConv(graph, x, w.conv2, h, wd));
  await stage("conv1", () => applyConv(graph, x, w.conv1, h, wd));
  for (const [i, m] of w.middle.entries()) await stage(`middle.${i}`, () => (m.kind === "attn" ? attnBlock(graph, m, x, h, wd) : resBlock(graph, m, x, h, wd)));
  for (const [i, u] of w.ups.entries()) {
    await stage(`up${i}`, () => {
      if (u.kind === "res") return resBlock(graph, u, x, h, wd);
      const c = u.conv.cin;
      const big = graph.buffer(h * wd * 4 * c * 4);
      graph.run(upsample2xWgsl, [graph.buffer(new Uint32Array([h, wd, c])), x, big], upsample2xWorkgroups(h, wd, c));
      [h, wd] = [h * 2, wd * 2];
      return applyConv(graph, big, u.conv, h, wd);
    });
  }
  await stage("head", () => applyConv(graph, silu(graph, pixelNorm(graph, x, w.headNorm, h * wd, w.head.cin), h * wd * w.head.cin), w.head, h, wd));
  return { image: x, height: h, width: wd, steps };
}
