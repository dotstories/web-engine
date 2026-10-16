import type { Graph } from "../gpu/graph";
import { toFloat32 } from "../weights/fp8";
import { readTensor, type Safetensors } from "../weights/safetensors";
import type { BlockWeights } from "./block";
import type { Linear } from "./ops";

/** Loads tensors from the person's model file straight into GPU buffers. Names are ComfyUI's. */

function tensor(st: Safetensors, name: string) {
  const t = st.tensors.get(name);
  if (!t) throw new Error(`Missing tensor ${name}: is this a Krea 2 file?`);
  return t;
}

/** An fp8 matrix [n, k] and its scalar weight_scale. */
export async function loadLinear(graph: Graph, file: Blob, st: Safetensors, name: string): Promise<Linear> {
  const w = tensor(st, `${name}.weight`);
  if (w.dtype !== "F8_E4M3" || w.shape.length !== 2) throw new Error(`${name}.weight: expected a 2D F8_E4M3 tensor, got ${w.dtype}`);
  const scale = tensor(st, `${name}.weight_scale`);
  const bytes = await readTensor(file, w);
  return {
    buffer: graph.buffer(new Uint32Array(bytes.buffer as ArrayBuffer, bytes.byteOffset, bytes.byteLength / 4)),
    scale: toFloat32(scale.dtype, await readTensor(file, scale))[0],
    n: w.shape[0],
    k: w.shape[1],
  };
}

/** Any float tensor as f32 on the CPU (norm weights, modulation tables are small). */
export async function loadFloats(file: Blob, st: Safetensors, name: string): Promise<Float32Array> {
  const t = tensor(st, name);
  return toFloat32(t.dtype, await readTensor(file, t));
}

export async function loadBlock(graph: Graph, file: Blob, st: Safetensors, index: number): Promise<BlockWeights> {
  const p = `blocks.${index}`;
  const lin = (name: string) => loadLinear(graph, file, st, `${p}.${name}`);
  const vec = async (name: string) => graph.buffer(await loadFloats(file, st, `${p}.${name}`));
  return {
    wq: await lin("attn.wq"),
    wk: await lin("attn.wk"),
    wv: await lin("attn.wv"),
    gate: await lin("attn.gate"),
    wo: await lin("attn.wo"),
    mlpGate: await lin("mlp.gate"),
    mlpUp: await lin("mlp.up"),
    mlpDown: await lin("mlp.down"),
    norm1: await vec("prenorm.scale"),
    norm2: await vec("postnorm.scale"),
    normQ: await vec("attn.qknorm.qnorm.scale"),
    normK: await vec("attn.qknorm.knorm.scale"),
    table: await loadFloats(file, st, `${p}.mod.lin`),
  };
}
